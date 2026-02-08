import os, json, time, re
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm


# ===== 依赖检查（友好提示）=====
try:
    from Bio import Entrez
except ImportError:
    raise ImportError("❌ 请先安装: pip install biopython")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("❌ 请先安装: pip install sentence-transformers")



class PubMedRetriever:
        
    def __init__(self, email: str = "xxx.com", cache_dir: str = "data/vectors"):
        """
        初始化
        
        参数:
            email: PubMed 要求提供邮箱（用于流量追踪，不会滥用）
            cache_dir: 向量缓存目录
        """
        self.email = email
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化嵌入模型（CPU 友好，70MB）
        print("加载 Sentence-BERT 模型 (首次需下载 ~70MB)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
        print("✅ 模型加载完成")
        
        # 生化知识库：常见靶点的 MeSH 术语映射
        self.target_mesh = {
            "egfr": '"Epidermal Growth Factor Receptor"[MeSH]',
            "her2": '"Receptor, ErbB-2"[MeSH]',
            "vegfr": '"Vascular Endothelial Growth Factor Receptor"[MeSH]',
            "pd-1": '"Programmed Cell Death 1 Receptor"[MeSH]',
            "ace2": '"Angiotensin-Converting Enzyme 2"[MeSH]',
        }
    
    def  _fetch_articles(self, query: str, max_results: int = 30) -> List[Dict]:
        """底层：执行 PubMed 检索 + 解析"""
        print(f"🔍 PubMed 检索: {query[:70]}...")
        
        try:
            # 检索 PMID
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, usehistory="y")
            results = Entrez.read(handle)
            id_list = results["IdList"]  # the structure of results?? # 匹配和获取的为什么还不一样？？
            print(f"匹配：{results['Count']} , 获取{len(id_list)}")
            if not id_list:
                return []
            # 获取详情

            handle = Entrez.efetch(db='pubmed', id=id_list, rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            articles = []
            for record in records["PubmedArticle"]:
                try: 
                    medline = record["MedlineCitation"]
                    article = medline["Article"]
                    pmid = medline["PMID"]
                    title = article["ArticleTitle"]
                    abstract = article.get("Abstract", {}).get("AbstractText", [""])[0]
                    if isinstance(abstract, list):
                        abstract = " ".join(str(seg) for seg in abstract)
                    
                    if not abstract or len(abstract) < 50:
                        continue
                    articles.append(                        
                        {"pmid": str(pmid),
                        "title": title,
                        "abstract": abstract,
                        "journal": article["Journal"]["Title"],
                        "year": article["Journal"]["JournalIssue"]["PubDate"].get("Year", "N/A"),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
                except Exception:
                    continue 
            print(f"一共找到{len(articles)}篇文章")

        except Exception as e:
            print(f"   ❌ PubMed 错误: {type(e).__name__}: {e}")
            return []

    def _embed_and_cache(self, articles: List[Dict], cache_key: str) -> np.ndarray:
        """向量化 + 缓存"""
        if not articles:
            return np.array([])
        
        print(f"🧠 向量化 {len(articles)} 篇摘要...")
        abstracts = [a["abstract"] for a in articles]
        embeddings = self.model.encode(abstracts, convert_to_numpy=True)
        
        # 保存缓存
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                "query": cache_key,
                "articles": articles,
                "embeddings": embeddings.tolist()
            }, f, ensure_ascii=False, indent=2)
        print(f"💾 缓存: {cache_file}")
        
        return embeddings
    

    def _load_cache(self, cache_key: str) -> (List[Dict], np.ndarray):
        """加载缓存"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if not os.path.exists(cache_file):
            return [], np.array([])
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data["articles"], np.array(data["embeddings"])
        except Exception:
            return [], np.array([])

        
       # ========== 模式 A：靶点驱动（Agent 自动调用）==========
    def retrieve_by_target(self, target: str, focus: str = "toxicity", top_k: int = 5) -> List[Dict]:
        """
        按靶点+焦点检索（适合 Agent 自动决策）
        
        示例:
            retriever.retrieve_by_target("EGFR", "cardiotoxicity")
            → 自动生成 Query: "EGFR AND cardiotoxicity AND humans[MeSH]"
        """
        # 构建缓存键（避免重复请求）
        cache_key = f"{target.lower()}_{focus.lower()}"
        
        # 尝试加载缓存
        articles, embeddings = self._load_cache(cache_key)
        if not articles:
            # 生成专业 Query（自由文本，避免 MeSH 陷阱）
            query = f"{target} AND {focus} AND humans[MeSH]"
            articles = self._fetch_articles(query, max_results=30)
            if not articles:
                return []
            embeddings = self._embed_and_cache(articles, cache_key)
        
        # 返回最新文献（按发表时间隐含排序）
        return articles[:top_k]


    def retrieve_by_query(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        按自然语言 Query 检索（适合用户提问）
        
        示例:
            retriever.retrieve_by_query("EGFR inhibitors QT prolongation arrhythmia")
            → 直接用该 Query 检索 + 语义重排
        """
        # 缓存键 = Query 哈希（相同 Query 复用结果）
        import hashlib
        cache_key = f"query_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        
        # 尝试加载缓存
        articles, embeddings = self._load_cache(cache_key)
        if not articles:
            # 直接用用户 Query 检索（不修改！）
            full_query = f"{query} AND humans[MeSH]"  # 仅追加人类研究限定
            articles = self._fetch_articles(full_query, max_results=30)
            if not articles:
                return []
            embeddings = self._embed_and_cache(articles, cache_key)
        
        # 语义重排（用户 Query 与摘要计算相似度）
        print(f"🎯 语义重排: '{query}'")
        query_emb = self.model.encode([query], convert_to_numpy=True)
        sims = np.dot(embeddings, query_emb.T).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        # 附加相似度分数
        results = []
        for idx in top_indices:
            art = articles[idx].copy()
            art["similarity"] = float(sims[idx])
            results.append(art)
        
        return results
            



    
if __name__ == "__main__":
    EMAIL = "Xin.Xu1@etu.univ-grenoble-alpes.fr"  # ← ← ← 替换为真实邮箱！
    retriever = PubMedRetriever(email=EMAIL)
    
    # 测试1：模式 A - 靶点驱动（Agent 自动调用）
    print("="*70)
    print("🧪 模式 A: 靶点驱动检索 (Agent 用)")
    print("   场景: Agent 决策时自动查询 'EGFR 心脏毒性'")
    print("="*70)
    results = retriever.retrieve_by_target("EGFR", "cardiotoxicity", top_k=2)
    for i, art in enumerate(results, 1):
        print(f"\n[{i}] PMID: {art['pmid']} | {art['year']} | {art['journal']}")
        print(f"    标题: {art['title'][:90]}...")
    
    # 测试2：模式 B - 用户驱动（自然语言提问）
    print("\n" + "="*70)
    print("🧪 模式 B: 用户驱动检索 (人机交互用)")
    print("   场景: 用户提问 '哪些 EGFR 抑制剂导致 QT 延长或心律失常？'")
    print("="*70)
    user_query = "EGFR inhibitors associated with QT prolongation or cardiac arrhythmia"
    results = retriever.retrieve_by_query(user_query, top_k=2)
    for i, art in enumerate(results, 1):
        print(f"\n[{i}] 相似度: {art['similarity']:.3f} | PMID: {art['pmid']}")
        print(f"    标题: {art['title'][:90]}...")
        print(f"    摘要: {art['abstract'][:120]}...")
    
    print("\n✅ 双模式测试完成！")
    print("\n💡 API 使用指南:")
    print("   • Agent 决策 → 用 retrieve_by_target(target, focus)")
    print("   • 用户提问   → 用 retrieve_by_query(natural_language_query)")

