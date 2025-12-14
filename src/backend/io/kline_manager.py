# Copyright 2025 jmake1
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import pandas as pd
from backend.brokers import get_broker
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from config import config

MAX_LIMIT_KLINES = 1000
MAX_WORKERS = config.parallel_workers


def paginate_backwards(fetch_func, symbol: str, limit: int, **kwargs) -> pd.DataFrame:

    all_dfs = []
    next_end = None

    while True:
        df = fetch_func(
            symbol=symbol,
            limit=limit,
            end_time=next_end,
            **kwargs
        )
        if df.empty:
            break
        df = df.sort_values('time')
        all_dfs.append(df)
        if len(df) < limit:
            break
        earliest = df['time'].iloc[0]
        next_end = int(earliest) - 1
        time.sleep(0.5)

    if not all_dfs:
        return pd.DataFrame()
    result = pd.concat(all_dfs, ignore_index=True)
    return result.sort_values('time').reset_index(drop=True)


def incremental_paginate_backwards(fetch_func, symbol: str, limit: int, last_timestamp: int, **kwargs) -> pd.DataFrame:
 
    all_dfs = []
    next_end = None

    while True:
        df = fetch_func(
            symbol=symbol,
            limit=limit,
            end_time=next_end,
            **kwargs
        )
        if df.empty:
            break
        df = df.sort_values('time')
        earliest = df['time'].iloc[0]

        if earliest > last_timestamp:
            all_dfs.append(df)
            if len(df) < limit:
                break
            next_end = int(earliest) - 1
            time.sleep(0.5)
            continue
        else:
            df_new = df[df['time'] > last_timestamp]
            if not df_new.empty:
                all_dfs.append(df_new)
            break

    if not all_dfs:
        return pd.DataFrame()
    result = pd.concat(all_dfs, ignore_index=True)
    return result.sort_values('time').reset_index(drop=True)


def fetch_full_symbol_data(client, symbol: str, instrument: str, interval: str) -> pd.DataFrame:

    klines = paginate_backwards(
        client.get_klines,
        symbol=symbol,
        limit=MAX_LIMIT_KLINES,
        interval=interval,
        instrument=instrument
    )
    if klines.empty:
        return pd.DataFrame()
    klines['symbol'] = symbol

    df = klines
    
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    df = df.drop(columns=['time'])
    cols = ['symbol', 'datetime'] + [c for c in df.columns if c not in ['symbol', 'datetime']]
    df = df[cols]

    for col in df.columns:
        if col not in ['symbol', 'datetime']:
            df[col] = df[col].astype('float64')
    return df


def fetch_incremental_symbol_data(client, symbol: str, last_timestamp: int, instrument: str, interval: str) -> pd.DataFrame:

    klines_inc = incremental_paginate_backwards(
        client.get_klines,
        symbol=symbol,
        limit=MAX_LIMIT_KLINES,
        last_timestamp=last_timestamp,
        interval=interval,
        instrument=instrument,
    )
    if klines_inc.empty:
        return pd.DataFrame()
    klines_inc['symbol'] = symbol
    df = klines_inc
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    df = df.drop(columns=['time'])
    cols = ['symbol', 'datetime'] + [c for c in df.columns if c not in ['symbol', 'datetime']]
    df = df[cols]

    for col in df.columns:
        if col not in ['symbol', 'datetime']:
            df[col] = df[col].astype('float64')
    return df


def update_klines(source: str, file_path: str, instrument: str, interval: str, return_data: bool=False, symbols: list=None):
    logger = logging.getLogger(__name__)

    try:
        existing_df = pd.read_feather(file_path)
        logger.info(f"Read {len(existing_df)} lines from {file_path}")
    except FileNotFoundError:
        logger.info(f"File {file_path} not found. Full fetching...")
        existing_df = pd.DataFrame(columns=['symbol','datetime'])

    for col in existing_df.columns:
        if col not in ['symbol', 'datetime']:
            existing_df[col] = existing_df[col].astype('float64')
            
    existing_df = existing_df[existing_df['datetime'].ne(existing_df.groupby('symbol')['datetime'].transform('max'))]

    grouped = existing_df.groupby('symbol')['datetime'].max().reset_index()
    grouped['last_timestamp_ms'] = grouped['datetime'].astype(int) // 10**6
    symbols_in_csv = set(grouped['symbol'].tolist())
    last_ts_map = dict(zip(grouped['symbol'], grouped['last_timestamp_ms']))
    
    client = get_broker(source)
    
    if symbols is not None and return_data:
        current_symbols = set(symbols)
        logger.info(f"Fetching data for {len(current_symbols)} symbols")
    else:
        all_current_symbols = set(client.get_symbols(instrument))
        current_symbols = {sym for sym in all_current_symbols if sym.endswith('USDT')}
        logger.info(f"{source} {instrument} currently supports {len(current_symbols)} USDT symbols")

    symbols_to_drop = symbols_in_csv - current_symbols
    if symbols_to_drop:
        logger.info(f"Discarding {len(symbols_to_drop)} currently not supported symbols {symbols_to_drop}")
        existing_df = existing_df[~existing_df['symbol'].isin(symbols_to_drop)].copy()
    else:
        logger.info("0 symbols to discard")

    symbols_to_add = current_symbols - symbols_in_csv
    if symbols_to_add:
        logger.info(f"{len(symbols_to_add)} symbols to add")
    else:
        logger.info("0 new symbols to add")

    symbols_to_update = symbols_in_csv & current_symbols
    logger.info(f"{len(symbols_to_update)} symbols to update")

    new_dfs = []
    inc_dfs = []

    count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}

        for sym in symbols_to_add:
            futures[executor.submit(fetch_full_symbol_data, client, sym, instrument, interval)] = sym

        for sym in symbols_to_update:
            last_ts = last_ts_map.get(sym)
            if last_ts is not None:
                futures[executor.submit(fetch_incremental_symbol_data, client, sym, last_ts, instrument, interval)] = sym

        for fut in as_completed(futures):
            sym = futures[fut]
            count+=1
            if sym in symbols_to_add:
                try:
                    df_full = fut.result()
                    if not df_full.empty:
                        new_dfs.append(df_full)
                        logger.info(f"Fetched {len(df_full)} lines (full) for {sym} | {count}")
                    else:
                        logger.info(f"No data returned for {sym} | {count}")
                except Exception as e:
                    logger.info(f"Error fetching full data {sym} | {count}: {e}")
            else:
                try:
                    df_inc = fut.result()
                    if not df_inc.empty:
                        inc_dfs.append(df_inc)
                        logger.info(f"Fetched {len(df_inc)} incremental lines for {sym} | {count}")
                    else:
                        logger.info(f"No new data for {sym} | {count}")
                except Exception as e:
                    logger.info(f"Error fetching incremental data for {sym} | {count}: {e}")

    combined_list = [existing_df]
    if new_dfs:
        combined_list.extend(new_dfs)
    if inc_dfs:
        combined_list.extend(inc_dfs)

    if combined_list:
        final_df = pd.concat(combined_list, ignore_index=True)
        final_df = final_df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
        for col in final_df.columns:
            if col not in ['symbol', 'datetime']:
                final_df[col] = final_df[col].astype('float64')
        if return_data:
            return final_df
        else:
            final_df.to_feather(file_path)
            logger.info(f"File updated in {file_path}, total {len(final_df)} lines")
    else:
        logger.info("No data, failed to update.")


