import pubchempy as pcp
import pandas as pd
import deepchem as dc
import os
import json
from tqdm import tqdm
import time

class LipinskiFilter:
    def __init__(self, cache_file = "pubchem_cache.json"):
        self.cache_file = cache_file
        self.cache = self.load_cache()
        self.rules = {
            "mw": ("分子量 ≤ 500 Da", "过大分子难以穿过细胞膜（类比：大分子蛋白质不易被肠道吸收）"),
            "logp": ("LogP ≤ 5", "脂溶性过高易在脂肪组织蓄积（类比：脂溶性维生素A/D过量中毒）"),
            "hbd": ("氢键供体 ≤ 5", "过多HBD降低膜通透性（类比：多羟基糖类难穿过血脑屏障）"),
            "hba": ("氢键受体 ≤ 10", "过多HBA增加水溶性但降低膜穿透"),
            "rotb": ("可旋转键 ≤ 10", "柔性过高降低靶点结合特异性")
        }
    
    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                return json.load(f)
        else:
            return {}

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    

    def fetch_properties(self, smiles_list, delay = 1.0):
        results = []
        new_fetches = 0
        for smiles in tqdm(smiles_list, desc="Progress"):
            #检查缓存
            if smiles in self.cache:
                results.append(self.cache[smiles])
                continue
            #调用PubChem API
            try:
                compounds = pcp.get_compounds(smiles, 'smiles')
                if compounds:
                    c = compounds[0]  # 为什么这里要取【0】位呢？
                    props = {
                        "simles": smiles,
                        "cid": c.cid, # cid 是什么？ 
                        "mw": float(c.molecular_weight) if c.molecular_weight else None,
                        "logp": float(c.xlogp) if c.xlogp else None,
                        "hbd": c.h_bond_donor_count,
                        "hba": c.h_bond_acceptor_count,
                        "rotb": c.rotatable_bond_count,
                        "tpsa": c.tpsa
                    }
                    self.cache[smiles] = props
                    results.append(props)
                    new_fetches += 1
                    time.sleep(delay)  # 防限流
                else:
                    results.append({"smiles": smiles, "error": "Not found"})
            except Exception as e:
                print(f"\n⚠️  {smiles[:20]}... 失败: {str(e)[:50]}")
                results.append({"smiles": smiles, "error": str(e)})
                time.sleep(delay * 2)
        
        if new_fetches > 0:
            self._save_cache()
            print(f"💾 已缓存 {new_fetches} 个新分子到 {self.cache_file}")
        
        return pd.DataFrame(results)

    def apply_rules(self, df):
        df = df.copy().dropna(subset=["mw", "logp", "hbd", "hba", "rotb"], how="all")
        
        df["pass_mw"] = df["mw"].fillna(999) <= 500
        df["pass_logp"] = df["logp"].fillna(99) <= 5
        df["pass_hbd"] = df["hbd"].fillna(99) <= 5
        df["pass_hba"] = df["hba"].fillna(99) <= 10
        df["pass_rotb"] = df["rotb"].fillna(99) <= 10
        df["lipinski_pass"] = (
            df["pass_mw"] & df["pass_logp"] & 
            df["pass_hbd"] & df["pass_hba"] & df["pass_rotb"]
        )
        
        def explain(row):
            if row["lipinski_pass"]:
                return "✅通过 Lipinski 五规则：具备口服生物利用度潜力"
            fails = []
            for col, (rule_name, _) in zip(
                ["pass_mw", "pass_logp", "pass_hbd", "pass_hba", "pass_rotb"],
                self.rules.values()
            ):
                if not row[col]:
                    fails.append(f"❌ {rule_name}")
            return " | ".join(fails[:2])
        
        df["explanation"] = df.apply(explain, axis=1)
        return df

    def filter_molecules(self, smiles_list):
        df = self.fetch_properties(smiles_list)
        df = self.apply_rules(df)
        
        passed = df[df["lipinski_pass"]].shape[0]
        total = len(smiles_list)
        print(f"\n📊 过滤结果: {passed}/{total} 个分子通过 ({passed/total*100:.1f}%)")
        return df
        


if __name__ == "__main__":
    

    # 临时测试脚本（保存为 test_imatinib.py）
    
    filter_tool = LipinskiFilter()
    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",      # 阿司匹林（通过）
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # 咖啡因（通过）
        "CCCCCCCCCCCCCCCCCC(=O)O",  # 伊马替尼（失败）
        "CCOC(=O)CC(N)C1=CC=C(O)C=C1",   # 左旋多巴（通过）
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # 布洛芬（通过）
    ]
    results = filter_tool.filter_molecules(test_smiles)
    
    # 保存结果
    results.to_csv("lipinski_results.csv", index=False)
    print("\n💾 结果已保存: lipinski_results.csv\n")
    
    # 打印简洁报告
    print("📋 简明报告:")
    for _, row in results.iterrows():
        status = "🟢 通过" if row.get("lipinski_pass", False) else "🔴 拒绝"
        cid = row.get('cid', 'N/A')
        mw = row.get('mw', 0)
        logp = row.get('logp', 0)
        print(f"{status} | CID:{str(cid):6} | MW:{mw:6.1f} | LogP:{logp:4.1f} | {row['explanation']}")
    

