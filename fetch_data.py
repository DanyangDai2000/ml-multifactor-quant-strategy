"""
fetch_data.py - 下载沪深300成分股 2018-2022 日线数据
使用 baostock 免费金融数据接口（无需 token，开源）

用法：
    source venv/bin/activate
    python fetch_data.py

输出：
    data/hs300_2018_2022.csv
"""

import os
import sys
import time
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_hs300_codes():
    """通过 baostock 获取沪深300成分股列表"""
    import baostock as bs
    
    print("[1/4] 登录 baostock...")
    lg = bs.login()
    if lg.error_code != "0":
        print(f"  登录失败: {lg.error_msg}")
        sys.exit(1)
    
    print("[2/4] 获取沪深300成分股列表...")
    rs = bs.query_hs300_stocks()
    if rs.error_code != '0':
        print(f"  查询失败: {rs.error_msg}")
        bs.logout()
        sys.exit(1)
    
    codes = []
    names = []
    while rs.next():
        row = rs.get_row_data()
        # row = [update_date, code, name]
        codes.append(row[1])    # sh.600000 格式
        names.append(row[2])    # 股票名称
    
    bs.logout()
    
    print(f"  获取到 {len(codes)} 只成分股")
    for i in range(min(5, len(codes))):
        print(f"    {codes[i]} - {names[i]}")
    print(f"    ... (共{len(codes)}只)")
    return codes


def fetch_daily_data(codes, start_date="2018-01-01", end_date="2022-12-31"):
    """批量下载日线数据"""
    import baostock as bs
    
    print(f"[3/4] 下载 {len(codes)} 只股票日线数据 ({start_date} ~ {end_date})...")
    
    bs.login()
    
    all_data = []
    total = len(codes)
    errors = 0
    
    for i, code in enumerate(codes):
        try:
            # baostock 的 code 格式类似 sh.600000
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,close,volume,peTTM,pbMRQ,turn,pctChg",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"  # 前复权
            )
            
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            
            if rows:
                df = pd.DataFrame(rows, columns=[
                    "date", "code", "close", "volume", "pe", "pb", 
                    "turnover", "pct_chg"
                ])
                # baostock code 是 sh.600000 格式，保留纯数字部分
                df["code"] = code.split(".")[-1]
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
                df["pe"] = pd.to_numeric(df["pe"], errors="coerce")
                all_data.append(df)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [WARN] {code} 失败: {e}")
        
        if (i + 1) % 30 == 0:
            print(f"  进度: {i+1}/{total} (成功{len(all_data)}只)")
        
        # 请求间隔，避免触发限流
        time.sleep(0.05)
    
    bs.logout()
    
    print(f"  下载完成: 成功{len(all_data)}只, 失败{errors}次")
    
    if not all_data:
        print("  [ERROR] 未下载到任何数据")
        return None
    
    result = pd.concat(all_data, ignore_index=True)
    print(f"  总行数: {len(result):,}")
    return result


def process_and_save(df):
    """清洗并保存"""
    print("[4/4] 清洗数据...")
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    
    # 添加行业分类（基于交易所板块）
    def guess_industry(code_str):
        s = str(code_str).strip().zfill(6)
        prefix = s[:3]
        if prefix in ["600", "601", "603", "605"]:
            return "主板-沪"
        elif prefix in ["000", "001", "002"]:
            return "主板-深"
        elif prefix == "300":
            return "创业板"
        elif prefix == "688":
            return "科创板"
        return "其他"
    
    df["industry"] = df["code"].astype(str).apply(guess_industry)
    
    # 只保留核心列
    keep = ["date", "code", "close", "volume", "pe", "industry"]
    have = [c for c in keep if c in df.columns]
    df = df[have]
    
    # 去空
    before = len(df)
    df = df.dropna(subset=["close", "pe"])
    print(f"  去空: {before} → {len(df)} 行")
    
    path = os.path.join(DATA_DIR, "hs300_2018_2022.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    
    print(f"  保存至: {path}")
    print(f"  维度: {df.shape}")
    print(f"  时间: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  股票: {df['code'].nunique()} 只")
    print(f"  列: {df.columns.tolist()}")
    
    return df


def main():
    codes = fetch_hs300_codes()
    raw = fetch_daily_data(codes)
    if raw is None:
        sys.exit(1)
    process_and_save(raw)
    print("\n✅ 数据下载完成！")
    print("现在可以运行: python main.py")


if __name__ == "__main__":
    main()
