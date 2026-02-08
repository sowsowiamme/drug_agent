# ===== 修复导入路径（添加到文件最顶部）=====
import sys
from pathlib import Path

# 自动定位项目根目录（向上查找包含 src/ 的目录）
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"🔧 已添加项目根目录到 sys.path: {project_root}")

import os, json, time
from typing import List, Dict
from src.filters.lipinski_filter import LipinskiFilter
from src.rag.pubmed_retriever import PubMedRetriever
from src.predictors.activity_predictor import ActivityPredictor
from src.utils.report_generator import ReportGenerator


class DrugScreeningAgent:
    """
    药物筛选决策 Agent
    
    决策逻辑（生化规则驱动）：
    1. Lipinski 过滤 → 淘汰理化性质不合格分子
    2. PubMed RAG → 检索靶点毒性文献
    3. 活性预测 → 评估靶点结合能力
    4. 综合评分 → 生成推荐报告
    
    优势：无需 LLM，用领域知识做可靠决策
    """
    
    def __init__(self, email: str, cache_dir: str = "data/cache"):
        self.email = email
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化三大模块
        print("🔧 初始化筛选模块...")
        self.lipinski_filter = LipinskiFilter(cache_file=os.path.join(cache_dir, "pubchem_cache.json"))
        self.pubmed_retriever = PubMedRetriever(email=email, cache_dir=os.path.join(cache_dir, "vectors"))
        self.activity_predictor = ActivityPredictor(cache_dir=cache_dir)
        self.report_generator = ReportGenerator()
        
        print("✅ Agent 初始化完成")

    def screen_molecule(self, smiles: str, name: str, target: str, focus: str = "toxicity") -> Dict:
        """
        筛选单个分子
        
        返回:
            完整决策报告（含各模块结果 + 综合评分）
        """
        report = {
            "name": name,
            "smiles": smiles,
            "target": target,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stages": {},
            "final_decision": None,
            "risk_score": 0.0,  # 0-100，越低越安全
            "recommendation": ""
        }
        
        print(f"\n🔬 筛选分子: {name} (靶点: {target})")
        print("-" * 60)
        
        # ===== 阶段 1: Lipinski 过滤 =====
        print("【阶段 1】Lipinski 理化性质过滤...")
        df = self.lipinski_filter.filter_molecules([smiles])
        if df.empty or not df.iloc[0]["lipinski_pass"]:
            report["stages"]["lipinski"] = {
                "passed": False,
                "explanation": df.iloc[0]["explanation"] if not df.empty else "数据获取失败",
                "properties": df.iloc[0].to_dict() if not df.empty else {}
            }
            report["final_decision"] = "REJECTED"
            report["risk_score"] = 90.0
            report["recommendation"] = "❌ 淘汰：理化性质不符合口服药物标准"
            print(f"   ❌ 淘汰: {report['stages']['lipinski']['explanation']}")
            return report
        
        # 提取理化性质
        props = df.iloc[0]
        report["stages"]["lipinski"] = {
            "passed": True,
            "mw": round(float(props["mw"]), 1),
            "logp": round(float(props["logp"]), 1),
            "hbd": int(props["hbd"]),
            "hba": int(props["hba"]),
            "rotb": int(props["rotb"])
        }
        print(f"   ✅ 通过: MW={props['mw']:.1f}, LogP={props['logp']:.1f}")
        
         # 阶段 2: PubMed RAG 毒性检索 
        print(f"\n【阶段 2】PubMed 文献检索 ({focus})...")
        # 策略：先用靶点驱动检索，再用用户 Query 重排
        articles = self.pubmed_retriever.retrieve_by_target(target, focus, top_k=3)
        
        if articles:
            # 检查高风险关键词
            high_risk_terms = ["fatal", "severe", "death", "withdrawn", "black box"]
            risk_count = sum(
                any(term in (a["title"] + a["abstract"]).lower() for term in high_risk_terms)
                for a in articles
            )
            
            report["stages"]["pubmed"] = {
                "articles_found": len(articles),
                "high_risk_count": risk_count,
                "articles": [
                    {
                        "pmid": a["pmid"],
                        "title": a["title"][:80] + "...",
                        "url": a["url"]
                    } for a in articles[:2]  # 仅存前2篇
                ]
            }
            # 风险评分
            if risk_count >= 2:
                report["risk_score"] += 40.0
                print(f"   ⚠️  高风险: {risk_count} 篇文献提示严重毒性")
            elif risk_count == 1:
                report["risk_score"] += 20.0
                print(f"   ⚠️  中风险: 1 篇文献提示潜在毒性")
            else:
                print(f"   ✅ 低风险: 未发现严重毒性报告")
        else:
            report["stages"]["pubmed"] = {"articles_found": 0, "high_risk_count": 0, "articles": []}
            print("   ℹ️  无相关文献（可能为新靶点）")

        # 阶段 3: 活性预测 
        print(f"\n【阶段 3】靶点活性预测...")
        activity_result = self.activity_predictor.predict_activity(smiles, target)
        report["stages"]["activity"] = activity_result
        
        pic50 = activity_result["pIC50"]
        if pic50 is None:
            report["risk_score"] += 30.0
            print("   ⚠️  活性预测失败")
        elif pic50 >= 8.0:
            print(f"   ✅ 高活性: pIC50={pic50}")
        elif pic50 >= 6.5:
            report["risk_score"] += 10.0
            print(f"   🟡 中等活性: pIC50={pic50}")
        else:
            report["risk_score"] += 25.0
            print(f"   🔴 低活性: pIC50={pic50}")

        # 综合决策 
        print("\n【综合决策】...")
        if report["risk_score"] >= 70.0:
            decision = "REJECTED"
            recommendation = "❌ 淘汰：综合风险过高（理化/毒性/活性任一环节失败）"
        elif report["risk_score"] >= 40.0:
            decision = "CAUTION"
            recommendation = "⚠️  谨慎推进：需额外毒理实验验证"
        else:
            decision = "RECOMMENDED"
            recommendation = "✅ 推荐：理化性质合格 + 无高风险毒性 + 活性良好"
        
        report["final_decision"] = decision
        report["recommendation"] = recommendation
        
        print(f"   风险评分: {report['risk_score']:.1f}/100")
        print(f"   决策: {recommendation}")
        
        return report

    def screen_batch(self, molecules: List[Dict], target: str, focus: str = "toxicity") -> List[Dict]:
        """
        批量筛选分子
        
        参数:
            molecules: [{"name": "阿司匹林", "smiles": "..."}, ...]
            target: 靶点名称
            focus: 毒性焦点
        
        返回:
            按推荐度排序的报告列表
        """
        print(f"🎯 批量筛选 {len(molecules)} 个分子 (靶点: {target})\n")
        
        reports = []
        for i, mol in enumerate(molecules, 1):
            print(f"\n[{i}/{len(molecules)}] {'='*50}")
            report = self.screen_molecule(mol["smiles"], mol["name"], target, focus)
            reports.append(report)
            time.sleep(0.5)  # 防 PubChem 限流
        
        # 按风险评分排序（低风险优先）
        reports.sort(key=lambda r: r["risk_score"])
        
        # 生成总结
        recommended = [r for r in reports if r["final_decision"] == "RECOMMENDED"]
        caution = [r for r in reports if r["final_decision"] == "CAUTION"]
        rejected = [r for r in reports if r["final_decision"] == "REJECTED"]
        
        summary = {
            "total": len(reports),
            "recommended": len(recommended),
            "caution": len(caution),
            "rejected": len(rejected),
            "top_recommendation": recommended[0] if recommended else None
        }
        
        return reports, summary


if __name__ == "__main__":
    EMAIL = "Xin.Xu1@etu.univ-grenoble-alpes.fr"
    
    # 测试分子集（4个通过 + 1个失败）
    test_molecules = [
        {"name": "吉非替尼", "smiles": "COCCN1CCN(CC1)Cc2ccc(cc2)NC(=O)c3cncc4ccccc34"},  # EGFR 抑制剂
        {"name": "布洛芬", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"},
        {"name": "阿司匹林", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
        {"name": "咖啡因", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
        {"name": "硬脂酸", "smiles": "CCCCCCCCCCCCCCCCCC(=O)O"}  # LogP 超标
    ]
    
    agent = DrugScreeningAgent(email=EMAIL)
    
    # 单分子测试
    print("="*70)
    print("🧪 Day 5 测试 1: 单分子筛选（吉非替尼）")
    print("="*70)
    report = agent.screen_molecule(
        smiles=test_molecules[0]["smiles"],
        name=test_molecules[0]["name"],
        target="EGFR",
        focus="cardiotoxicity"
    )
    
    # 批量测试
    print("\n" + "="*70)
    print("🧪 Day 5 测试 2: 批量筛选（5 个分子）")
    print("="*70)
    reports, summary = agent.screen_batch(
        molecules=test_molecules,
        target="EGFR",
        focus="toxicity"
    )
    
    print("\n" + "="*70)
    print("📊 筛选总结")
    print("="*70)
    print(f"   总分子数: {summary['total']}")
    print(f"   ✅ 推荐: {summary['recommended']}")
    print(f"   ⚠️  谨慎: {summary['caution']}")
    print(f"   ❌ 淘汰: {summary['rejected']}")
    
    if summary["top_recommendation"]:
        top = summary["top_recommendation"]
        print(f"\n🏆 首选分子: {top['name']}")
        print(f"   风险评分: {top['risk_score']:.1f}/100")
        print(f"   pIC50: {top['stages']['activity']['pIC50']}")
        print(f"   决策: {top['recommendation']}")
    
    print("\n✅ Day 5 Agent 筛选模块测试完成！")
