#!/usr/bin/env python
# src/utils/report_generator.py

import os,json
from datetime import datetime

class ReportGenerator:
    def __init__(self):
        pass
    
    def generate_markdown(self, report: dict, output_file: str = None) -> str:
        """生成 Markdown 报告"""
        md = []
        md.append("# 药物筛选综合报告")
        md.append(f"**分子**: {report['name']}")
        md.append(f"**靶点**: {report['target']}")
        md.append(f"**时间**: {report['timestamp']}")
        md.append("")
        
        # 阶段 1: Lipinski
        md.append("## 1. Lipinski 理化性质筛选")
        lip = report["stages"]["lipinski"]
        if not lip.get("passed", False):
            md.append(f"❌ **淘汰**: {lip.get('explanation', 'N/A')}")
        else:
            md.append("✅ **通过**")
            md.append(f"- 分子量 (MW): {lip['mw']} Da (阈值 ≤500)")
            md.append(f"- 脂水分配系数 (LogP): {lip['logp']} (阈值 ≤5)")
            md.append(f"- 氢键供体 (HBD): {lip['hbd']} (阈值 ≤5)")
            md.append(f"- 氢键受体 (HBA): {lip['hba']} (阈值 ≤10)")
            md.append(f"- 可旋转键 (RotB): {lip['rotb']} (阈值 ≤10)")
        
        # 阶段 2: PubMed
        md.append("\n## 2. 毒性文献检索")
        pub = report["stages"]["pubmed"]
        if pub["articles_found"] == 0:
            md.append("ℹ️  未检索到相关文献（可能为新靶点或新化学空间）")
        else:
            md.append(f"📚 检索到 {pub['articles_found']} 篇文献")
            if pub["high_risk_count"] > 0:
                md.append(f"⚠️  **高风险提示**: {pub['high_risk_count']} 篇文献含严重毒性关键词")
            for i, art in enumerate(pub["articles"], 1):
                md.append(f"{i}. [{art['pmid']}] {art['title']}")
                md.append(f"   [PubMed 链接]({art['url']})")
        
        # 阶段 3: 活性
        md.append("\n## 3. 靶点活性预测")
        act = report["stages"]["activity"]
        if act["pIC50"] is None:
            md.append("⚠️  活性预测失败")
        else:
            md.append(f"**预测 pIC50**: {act['pIC50']} (Confidence: {act['confidence']})")
            md.append(f"\n**生化解说**:\n```\n{act['explanation']}\n```")
        
        # 综合决策
        md.append("\n## 4. 综合决策")
        md.append(f"**风险评分**: {report['risk_score']:.1f}/100")
        md.append(f"**最终决策**: {report['recommendation']}")
        
        # 生化建议（你的差异化优势！）
        md.append("\n## 5. 营养学视角建议")
        if "吡啶环" in str(report):
            md.append("💡 **吡啶环类比**: 类似维生素 B3（烟酸）结构，参与 NAD+/NADH 氧化还原循环")
            md.append("   → 建议监测肝酶（ALT/AST），因烟酸高剂量可致肝损伤")
        if "氟原子" in str(report):
            md.append("💡 **氟原子类比**: 类似饮用水氟化物防龋机制")
            md.append("   → 长期使用需监测骨密度（氟蓄积风险）")
        if report['risk_score'] > 50:
            md.append("⚠️  **营养干预建议**: 高风险分子建议联用抗氧化剂（如维生素 E/C）")
            md.append("   → 类比：化疗药物联用抗氧化剂减轻氧化应激损伤")
        
        md_text = "\n".join(md)
        
        if output_file:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md_text)
            print(f"💾 报告已保存: {output_file}")
        
        return md_text
    
    def generate_summary(self, reports: list, summary: dict, output_file: str = None) -> str:
        """生成批量筛选总结报告"""
        md = []
        md.append("# 批量药物筛选总结报告")
        md.append(f"**靶点**: {reports[0]['target'] if reports else 'N/A'}")
        md.append(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"**总分子数**: {summary['total']}")
        md.append("")
        md.append(f"| 决策 | 数量 | 比例 |")
        md.append(f"|------|------|------|")
        md.append(f"| ✅ 推荐 | {summary['recommended']} | {summary['recommended']/summary['total']*100:.0f}% |")
        md.append(f"| ⚠️  谨慎 | {summary['caution']} | {summary['caution']/summary['total']*100:.0f}% |")
        md.append(f"| ❌ 淘汰 | {summary['rejected']} | {summary['rejected']/summary['total']*100:.0f}% |")
        md.append("")
        
        if summary["top_recommendation"]:
            top = summary["top_recommendation"]
            md.append("## 🏆 首选分子")
            md.append(f"**{top['name']}**")
            md.append(f"- 风险评分: {top['risk_score']:.1f}/100")
            md.append(f"- pIC50: {top['stages']['activity']['pIC50']}")
            md.append(f"- 决策: {top['recommendation']}")
        
        md_text = "\n".join(md)
        
        if output_file:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md_text)
            print(f"💾 总结报告已保存: {output_file}")
        
        return md_text

# ===== 测试 =====
if __name__ == "__main__":
    # 模拟报告数据
    sample_report = {
        "name": "吉非替尼",
        "smiles": "COCCN1CCN(CC1)Cc2ccc(cc2)NC(=O)c3cncc4ccccc34",
        "target": "EGFR",
        "timestamp": "2026-02-08 21:00:00",
        "stages": {
            "lipinski": {
                "passed": True,
                "mw": 446.9,
                "logp": 3.8,
                "hbd": 2,
                "hba": 7,
                "rotb": 8
            },
            "pubmed": {
                "articles_found": 3,
                "high_risk_count": 1,
                "articles": [
                    {"pmid": "36453210", "title": "Osimertinib-associated cardiotoxicity...", "url": "https://pubmed.ncbi.nlm.nih.gov/36453210/"},
                    {"pmid": "35129488", "title": "Gefitinib-induced cardiotoxicity...", "url": "https://pubmed.ncbi.nlm.nih.gov/35129488/"}
                ]
            },
            "activity": {
                "pIC50": 8.45,
                "confidence": 0.89,
                "explanation": "🌟 高活性（pIC50 ≥ 8.0）\n   预测依据: 预测对靶点有强结合能力",
                "key_substructures": ["吡啶环", "氟原子"]
            }
        },
        "final_decision": "RECOMMENDED",
        "risk_score": 35.0,
        "recommendation": "✅ 推荐：理化性质合格 + 无高风险毒性 + 活性良好"
    }
    
    generator = ReportGenerator()
    md = generator.generate_markdown(sample_report, "data/outputs/sample_report.md")
    print("✅ Markdown 报告生成成功！预览前 10 行:")
    print("\n".join(md.split("\n")[:10]))