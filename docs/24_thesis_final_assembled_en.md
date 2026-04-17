# Personalization of Customer Offers Based on Banking Services User Transaction Activity

## ABSTRACT
This thesis studies the problem of personalizing banking offers from user transaction activity under restricted access to real banking data. The work builds a reproducible recommendation pipeline that combines generation of a banking-like synthetic dataset, exploratory data analysis, training of baseline and advanced recommendation models, offline evaluation with top-K ranking metrics, and a lightweight service prototype for recommendation delivery. The implemented experimental stack includes profile similarity, item-kNN, implicit matrix factorization, Neural Collaborative Filtering, LightGCN, SASRec, and a time-decay model that emphasizes recency of user interests. In addition, the study includes bootstrap analysis, a multi-seed benchmark, and minimal transfer validation on the real Online Retail transaction log. The results show that time-aware behavioral modeling remains the strongest approach in the synthetic setting, while the real-data benchmark confirms that the pipeline transfers to real transaction logs even though the strongest model there is a simpler implicit MF baseline. The practical outcome is a reproducible research and software contour that can be extended toward real banking deployments.

## KEYWORDS
bank offer personalization; recommender systems; transaction data; implicit feedback; top-K ranking; Neural Collaborative Filtering; LightGCN; SASRec; synthetic data; FastAPI

## АННОТАЦИЯ
В работе рассматривается задача персонализации клиентских предложений банка на основе транзакционной активности пользователя в условиях ограниченного доступа к реальным банковским данным. Цель исследования состоит в построении воспроизводимого recommendation pipeline, который объединяет генерацию banking-like синтетического датасета, разведочный анализ данных, обучение baseline- и advanced-моделей, офлайн-оценку качества по метрикам top-K ранжирования и сервисный прототип выдачи рекомендаций. В экспериментальную часть включены profile similarity, item-kNN, implicit matrix factorization, Neural Collaborative Filtering, графовая модель LightGCN и sequence-aware модель SASRec, а в качестве модели расширенного уровня предложен time-decay подход, учитывающий свежесть интересов пользователя. Дополнительно проведены bootstrap-анализ, multi-seed benchmark и минимальная валидация на реальном транзакционном retail-датасете Online Retail. Практический результат работы заключается в создании воспроизводимого исследовательского контура и рабочего API-прототипа, пригодного для дальнейшего расширения под реальные банковские данные.

## КЛЮЧЕВЫЕ СЛОВА
персонализация клиентских предложений; рекомендательные системы; транзакционные данные; банковские офферы; implicit feedback; top-K ранжирование; Neural Collaborative Filtering; LightGCN; SASRec; synthetic data; FastAPI

## List of Abbreviations
- `API` - application programming interface.
- `AUC` - area under the ROC curve.
- `CF` - collaborative filtering.
- `CTR` - click-through rate.
- `EDA` - exploratory data analysis.
- `MCC` - merchant category code.
- `MAP@K` - mean average precision at top-K.
- `MF` - matrix factorization.
- `MLOps` - machine learning operations.
- `NDCG@K` - normalized discounted cumulative gain at top-K.
- `NLP` - natural language processing.
- `P@K` - precision at top-K.
- `R@K` - recall at top-K.
- `SOTA` - state of the art.
- `TF-IDF` - term frequency-inverse document frequency.

## 1. Introduction
The digital transformation of financial services has changed the way banks communicate with clients. A modern banking ecosystem no longer consists only of a current account and a payment card. It usually includes investment products, deposits, insurance services, installment programs, credit products, partner subscriptions, educational services, loyalty programs, and marketplace-like extensions. As the number of available products increases, the quality of personalization becomes strategically important. If the bank recommends irrelevant products, communication fatigue grows, conversion falls, and the client starts perceiving outreach as spam rather than assistance. If, in contrast, the bank is able to identify products that are genuinely relevant to current user needs, the recommendation becomes a form of contextual service rather than advertising.

One of the richest signals available to a bank is transaction history. Transactions reflect actual behavior rather than self-reported intentions. They contain evidence about what the client spends money on, how often purchases are made, which categories dominate monthly behavior, and how stable or volatile the spending pattern is over time. For example, regular spending on groceries, pharmacies, utilities, and children's goods may indicate a family-oriented segment. Frequent payments in travel, transport, hotels, and restaurants may indicate a travel-heavy lifestyle or high mobility. Repeated investment-related or money transfer operations may suggest financial awareness and interest in brokerage or savings products. From the perspective of recommender systems, such behavioral traces can be transformed into a user representation and matched against candidate offers.

The practical relevance of this problem is especially high in banking because financial institutions operate under strong data constraints and high expectations regarding responsible personalization. On the one hand, banks usually possess structured and frequent behavioral data, which is an advantage compared with many retail domains. On the other hand, those data are sensitive, access is restricted, and public reproduction of industrial experiments is difficult. This creates a gap between academic recommendation research and realistic banking personalization tasks. Many studies demonstrate strong recommendation models, but fewer works provide a fully reproducible pipeline that approximates the banking setting under open conditions.

This thesis addresses that gap by constructing and evaluating a reproducible recommendation pipeline for personalization of banking offers from transaction activity. The central design choice is to use synthetic data rather than real client data. This choice is not a weakness to be hidden, but a methodological response to privacy and access limitations. The use of synthetic data makes it possible to build an end-to-end system, compare models under a stable protocol, perform robustness analysis, and demonstrate a service prototype without requiring privileged access to proprietary banking datasets.

The research goal of the thesis is to develop and evaluate a recommendation pipeline that ranks banking offers for a user based on transaction-derived behavior. The work combines ideas from recommender systems, behavioral feature engineering, semantic representation of offers, and service-oriented implementation.

The research objectives are the following:
1. analyze relevant literature on transaction-based personalization, ranking metrics, and modern recommender architectures;
2. design a synthetic dataset that mimics a banking personalization scenario under realistic privacy constraints;
3. implement and compare several recommendation approaches within a unified top-K evaluation framework;
4. study the robustness of conclusions through segment-wise, bootstrap, and multi-seed analysis;
5. demonstrate the practical deployability of the approach through a simple API service prototype.

The thesis is guided by the following research questions:
1. Are transaction-derived features informative enough to support meaningful personalization of banking offers?
2. Which simple and moderately advanced recommendation strategies are most competitive in the current synthetic setting?
3. How stable are the observed model differences under repeated evaluation and uncertainty analysis?
4. Which elements of the current pipeline appear most promising for further development toward a more realistic production-grade recommender?

The working hypothesis of the study is that transaction-driven user representations can support useful top-K ranking of banking offers, and that temporal weighting or lightweight semantic augmentation may improve ranking quality over a purely static profile-matching baseline. At the same time, the thesis does not assume that the most sophisticated model will necessarily dominate in a synthetic environment. The study explicitly tests whether additional complexity is justified by measurable gains.

The scientific contribution of the work lies in the systematic comparison of multiple recommender approaches in a reproducible banking-inspired setup with explicit uncertainty analysis. The practical contribution lies in the creation of a full pipeline that includes data generation, model evaluation, reporting artifacts, and a service prototype. Even if the synthetic environment limits external validity, the thesis provides a concrete methodological scaffold that can later be transferred to a real-data industrial setting.

The remainder of the thesis is organized as follows. Chapter 2 reviews related literature and positions the study within recommender systems and banking personalization research. Chapter 3 describes the synthetic dataset design and presents exploratory analysis of the generated data. Chapter 4 formalizes the ranking problem and evaluation protocol. Chapter 5 explains the implemented methods and engineering decisions. Chapter 6 presents experiments and results. Chapter 7 describes the service prototype and deployment considerations. Chapter 8 discusses interpretation, practical implications, limitations, and future work. Chapter 9 concludes the thesis. The appendices provide additional implementation, literature, and reproducibility details.

## 2. Related Work
### 2.1 Recommendation in Implicit-Feedback Settings
Recommendation problems are often formulated in two broad ways: explicit-feedback recommendation and implicit-feedback recommendation. In explicit-feedback settings, the model predicts ratings or some directly expressed user preference. In implicit-feedback settings, the system must infer preference from behavior such as clicks, purchases, watch time, or transactions. Banking personalization belongs much more naturally to the second family. A client rarely provides explicit ratings for banking products, yet transaction behavior generates abundant implicit evidence about interests and needs.

The literature on implicit-feedback recommendation emphasizes ranking rather than pointwise prediction. What matters is not whether the model can assign an absolute score to every product, but whether the top positions of the recommendation list contain items that are most relevant for the user. This focus motivates metrics such as `Precision@K`, `Recall@K`, `MAP@K`, and `NDCG@K`, which are especially suitable when the application interface displays only a small number of offers. In a banking application, the difference between rank 1 and rank 5 is meaningful, while the exact ordering among irrelevant offers is not.

This thesis follows the implicit-feedback ranking tradition because banking transactions provide exactly this kind of signal. The evaluation protocol, holdout design, and model comparison all reflect the practical objective of retrieving a short ranked list rather than estimating a dense utility function over every possible user-offer pair.

### 2.2 Collaborative Filtering and Neural Interaction Models
Classical collaborative filtering methods model preference from user-item interactions rather than from hand-crafted features. Matrix factorization became a foundational approach because it projects users and items into a shared latent space and estimates affinity through their interaction. Although conceptually simple, this approach remains a strong baseline in many domains. In industrial settings, collaborative filtering is often valued because it can discover patterns that are not directly visible in explicit item metadata.

Neural Collaborative Filtering by He et al. [1] extended this line of research by replacing the simple inner product between latent user and item vectors with a nonlinear matching function. The key argument of the paper is that latent interactions can be richer than what linear factorization captures. This work is important for the present thesis for two reasons. First, it formalizes a standard comparison point for implicit-feedback recommendation. Second, it shows how recommendation quality can sometimes improve when the model is allowed to learn more flexible interaction patterns.

At the same time, collaborative filtering approaches depend strongly on the density and richness of the interaction graph. When the number of items is small, when interactions are synthetically generated from explicit category logic, or when user-item observations are limited, latent-factor methods may underperform simpler content-aligned baselines. This thesis includes an implicit matrix factorization baseline precisely to test this issue empirically rather than assume it in advance.

### 2.3 Sequential Recommendation
A limitation of static collaborative filtering is that it often does not represent the order of user actions. In many real applications, recent behavior is more informative than older history. Sequential recommender systems address this by modeling event order and, in some cases, attention over previous actions. SASRec by Kang and McAuley [2] showed that self-attention is a powerful mechanism for modeling user behavior sequences. The method captures which previous items are most informative for the next interaction and can adapt to both short and long sequences.

BERT4Rec by Sun et al. [3] further advanced sequential recommendation by adapting bidirectional Transformer representations to the recommendation domain. Instead of using only left-to-right next-item prediction, the model uses a masked objective that learns contextual dependencies from both sides of the sequence. This design often improves representation quality when enough sequential data are available.

Sequential models are directly relevant to transaction-based banking personalization because transactions are inherently time-ordered. A user may show temporary interests, seasonal spending, or life-event-driven shifts. For example, a spike in education-related payments may reflect course enrollment; a cluster of travel-related spending may signal near-term mobility; a sudden increase in home-improvement purchases may indicate a renovation phase. Such signals are poorly captured by a purely aggregate user profile. In the current project, the main serving story still relies on lighter models, but a compact SASRec branch was also implemented on the real transaction log as an auxiliary sequence-aware validation. This makes the sequential direction part of the practical thesis scope rather than only a literature note.

### 2.4 Graph-Based Recommendation
Another major family of modern recommenders treats the interaction system as a graph. Users, items, merchants, or auxiliary entities become nodes, while interactions and semantic relations become edges. LightGCN by He et al. [4] simplified earlier graph-convolution approaches and demonstrated that a lighter message-passing design can remain highly competitive for recommendation tasks. In the present project, a simplified LightGCN branch was implemented as a literature-oriented graph baseline. It does not become the strongest model in the current experiments, but it demonstrates that graph recommendation can be integrated into the same reproducible pipeline.

Graph-based reasoning is particularly attractive in transaction data because many relationships are relational by nature. Users are linked to categories, merchants, locations, and product families. However, graph methods also require careful design of graph structure, edge semantics, and computational resources. The current thesis therefore treats LightGCN as an auxiliary advanced branch rather than the main serving model, while still keeping richer graph modeling as a well-motivated future direction.

### 2.5 Feature-Interaction and Hybrid Ranking Models
Industrial recommendation systems often rely on hybrid models that combine memorization of frequent patterns with generalization to new combinations of features. Wide & Deep [5] is a representative example of this approach. The wide component can learn sparse cross-feature patterns, while the deep component learns distributed generalizations. In click and conversion tasks, this mixture is often more effective than either purely shallow or purely deep methods.

DeepFM [6] and xDeepFM [7] further developed the feature-interaction paradigm by modeling low-order and high-order interactions more directly. These models are especially relevant when the application contains rich categorical and numerical features, such as user segment, transaction channel, category distribution, and offer type. Although the present thesis does not implement a full feature-interaction ranking network, the design of the recommendation problem leaves room for such a next step. The current data schema, segment definitions, and offer metadata were designed so that richer tabular recommendation models could be added without redesigning the full pipeline.

### 2.6 Semantic Similarity and NLP-Based Offer Representation
Offer metadata are usually textual or semi-textual. Product descriptions, campaign texts, category tags, and marketing explanations all contain semantic information. Sentence-BERT by Reimers and Gurevych [8] showed that sentence embeddings can be efficiently used for semantic similarity tasks. While the present thesis implements a lightweight TF-IDF semantic layer rather than a transformer embedding model, the conceptual role is the same: offer text can be transformed into a vector representation and compared with a representation of user interests.

This is especially important because banking offers are not just category labels. Two products may target similar categories but differ in value proposition. For example, a travel card and an insurance bundle may both be relevant to a frequent traveler, but the textual framing of benefits differs. Similarly, an education subscription and an investment starter account may both be relevant to a financially curious student, but for different reasons. A semantic representation can capture these nuances more flexibly than a manual category overlap.

### 2.7 Transaction-Based Personalization in the Financial Domain
Several domain-specific works demonstrate the relevance of recommendation from transaction data. Pcard by Yi et al. [9] focused on personalized restaurant recommendation from payment card transactions. The study is especially important because it shows that transaction sequences can support recommendation even without the typical retail clickstream setting. Biswas et al. [10] explored recommendation through link prediction on transactional data, showing that graph-based structures can support merchant recommendation. Baek et al. [11] studied merchant recommendation using credit card payment data and showed the importance of domain-specific features.

Another especially relevant reference is the work on adaptive collaborative filtering with personalized time decay for financial product recommendation [12]. This paper provides strong domain motivation for emphasizing recency in financial recommendation. In many banking scenarios, a customer's needs change over time, and the freshest transaction evidence may better reflect short-term intent than long-run averages.

### 2.8 Synthetic Data in Financial Machine Learning
Access to real financial data is notoriously difficult due to privacy, legal, and commercial constraints. This makes reproducibility a major challenge. Boullé et al. [13] and related work on synthetic credit card transaction generation show that transaction-like data can be generated for research and testing. Synthetic data do not replace real validation, but they enable controlled experimentation, pipeline debugging, and open academic comparison.

For the purposes of this thesis, synthetic data serve four distinct roles. First, they remove the access barrier. Second, they allow transparent documentation of the generating assumptions. Third, they make repeated experiments reproducible because all random seeds and generation rules are controlled. Fourth, they enable service prototyping without exposing sensitive information.

### 2.9 Research Gap and Positioning of the Thesis
The literature review suggests several important conclusions. First, implicit-feedback ranking is the right general framing for the problem. Second, temporal and sequential structure matter in transaction-based recommendation. Third, semantic information from offer descriptions is potentially valuable. Fourth, real-data banking studies are informative but often impossible to reproduce directly in an open academic setting.

This thesis positions itself at the intersection of those observations. It does not claim a new SOTA recommendation architecture. Instead, it contributes a reproducible comparative study in a banking-inspired environment. The focus is on methodical construction of the pipeline, transparent model comparison, uncertainty analysis, and practical applicability through a service prototype. This positioning is academically defensible because reproducibility, careful evaluation, and clear limitation analysis are valuable contributions when real-data access is unavailable.

## 3. Data Design and Exploratory Analysis
### 3.1 Why Synthetic Data Were Used
The project topic naturally suggests the use of real banking transactions. However, such data are practically inaccessible in an academic context due to privacy, regulation, institutional policy, and the high cost of secure handling. Even when a bank exposes open information about product catalogs or generalized category lists, individual-level transaction histories are not available for research reproduction. Therefore, the present work adopts a synthetic-data strategy.

The rationale for using synthetic data is threefold. First, it allows the thesis to remain implementable under realistic time and access constraints. Second, it supports full reproducibility: any researcher can regenerate the dataset and rerun the same experiments. Third, it makes assumptions explicit. In many industrial studies, the researcher works with a dataset whose exact biases are only partially documented. In a synthetic setup, the generating assumptions are known and can be analyzed directly.

The key methodological requirement was not to create arbitrary random transactions, but to generate a dataset that is coherent enough for meaningful recommendation experiments. This means that user segments, transaction categories, amount distributions, and offer interactions should all follow interpretable patterns.

### 3.2 Data Generation Pipeline
The synthetic pipeline creates four related artifacts:
1. a user table with segment labels and user-level attributes;
2. a transaction table with user identifier, timestamp, category, amount, and behavioral context;
3. an offer catalog with product type, target categories, and textual description;
4. a user-offer interaction table for offline recommendation evaluation.

The implementation is deterministic under a fixed random seed. This property is crucial because it makes benchmark comparisons and future thesis extensions traceable. The main reference run of the project uses 800 users, 112,626 transactions, 15 offers, and 6,442 interactions. The dataset spans approximately one year, from March 14, 2025 to March 14, 2026.

[[FIGURE: Structure of the synthetic dataset: users, transactions, offers, and interactions]]

### 3.3 User Segmentation Logic
User behavior was designed around six synthetic segments:
1. `daily_life`;
2. `digital_pro`;
3. `family`;
4. `investor`;
5. `student`;
6. `traveler`.

These segments are not intended to represent real demographic truth. Their purpose is to create meaningful variation in category preferences and to make later segment-wise evaluation interpretable. In the reference run, the segment distribution is as follows:

| Segment | User Count | Interpretation |
|---|---:|---|
| daily_life | 205 | stable everyday spending with focus on groceries, transport, utilities, and routine services |
| digital_pro | 147 | higher share of electronics, digital subscriptions, and transfers |
| family | 162 | household-oriented behavior with groceries, home, healthcare, and utilities |
| investor | 80 | above-average interest in investments and money transfers |
| student | 99 | education, transport, entertainment, and budget-sensitive daily behavior |
| traveler | 107 | travel, transport, restaurants, and mobility-linked spending |

The segment distribution was chosen so that no single group dominates the entire dataset, while still reflecting plausible differences in scale. For example, the `investor` segment is intentionally smaller than `daily_life`, since a narrowly finance-oriented lifestyle is usually less frequent than routine spending behavior.

[[FIGURE: Distribution of users across behavioral segments]]

### 3.4 Transaction Schema
Each synthetic transaction contains a user identifier, timestamp, spending category, amount, and channel-like contextual information. The category system is MCC-inspired rather than a direct one-to-one copy of public MCC tables. It abstracts over a set of financially meaningful domains:
- groceries;
- restaurants;
- transport;
- travel;
- education;
- entertainment;
- electronics;
- investments;
- money_transfer;
- utilities;
- home;
- healthcare;
- insurance;
- fashion;
- cash_withdrawal.

This abstraction is sufficient for recommendation experiments because the goal is not MCC code prediction but behavior summarization for personalization. Using broader categories also makes the resulting model outputs easier to interpret in the thesis.

### 3.5 Offer Catalog Design
The offer catalog contains 15 synthetic banking or partner products. The catalog intentionally mixes core banking products and ecosystem-style services. Examples include `Smart Daily Cashback Card`, `Travel Miles Premium Card`, `Beginner Investment Account`, `Flexible Personal Loan`, `Family Savings Deposit`, `EdTech Learning Subscription`, and `Balanced Finance Bundle`.

The diversity of the catalog is important for two reasons. First, it prevents the recommendation problem from collapsing into only one product type, such as cards. Second, it enables the project to stay aligned with the original theme of personalization of customer offers in a broader banking ecosystem. Some products are clearly bank-native, such as deposits, loans, or insurance bundles. Others, such as learning subscriptions, represent ecosystem or partner offers, which were part of the user's initial project idea.

The following table summarizes the catalog at a high level:

| Offer ID | Offer Name | Type | Target Categories |
|---|---|---|---|
| O001 | Smart Daily Cashback Card | card | groceries, restaurants, transport |
| O002 | Travel Miles Premium Card | card | travel, transport, restaurants |
| O003 | Beginner Investment Account | investment | investments, money_transfer, education |
| O004 | Flexible Personal Loan | credit | home, electronics, fashion |
| O005 | Mortgage Refinance Program | credit | home, utilities, insurance |
| O006 | Health and Family Insurance Bundle | insurance | healthcare, insurance, home |
| O007 | Auto and Mobility Protection | insurance | transport, insurance, travel |
| O008 | Family Savings Deposit | deposit | groceries, home, utilities |
| O009 | EdTech Learning Subscription | partner | education, electronics, money_transfer |
| O010 | Premium Lifestyle Subscription | subscription | entertainment, restaurants, fashion |
| O011 | Utility AutoPay Cashback | card | utilities, home, money_transfer |
| O012 | Digital Security Package | service | electronics, money_transfer, investments |
| O013 | Student Smart Start Card | card | education, transport, entertainment |
| O014 | Cash Management Plus | service | cash_withdrawal, money_transfer, groceries |
| O015 | Balanced Finance Bundle | bundle | investments, insurance, utilities |

### 3.6 Interaction Simulation
The central evaluation object is not the transaction table itself, but the user-offer interaction table used for ranking metrics. Those interactions are generated from compatibility between user behavior and offer intent, with stochastic variation so that the benchmark is not entirely deterministic. This means that an offer is more likely to receive a positive interaction from users whose transaction history aligns with its target categories, but randomness preserves noise and prevents a trivial one-to-one mapping.

The interaction simulator was designed to satisfy several constraints:
1. positive events should be sufficiently frequent for offline ranking evaluation;
2. no single offer should dominate the whole benchmark;
3. user preferences should remain heterogeneous across segments;
4. the recommendation task should still require ranking rather than degenerate into exact label matching.

In the reference data, the positive interaction rate is approximately 0.202. The most frequently positive offers are O009, O012, O013, O001, and O002, each with around ninety positive interactions. This is useful because it produces a balanced-enough benchmark without creating an extreme popularity skew.

### 3.7 Dataset Statistics
Exploratory analysis of the generated data confirms that the dataset is nontrivial but well-behaved for experimentation. Key statistics are summarized below:

| Statistic | Value |
|---|---:|
| Users | 800 |
| Transactions | 112,626 |
| Offers | 15 |
| User-offer interactions | 6,442 |
| Positive interaction rate | 0.2020 |
| Average transactions per user | 140.78 |
| Median transactions per user | 141.00 |
| Average total spend per user | 1,158,123.26 |
| Median total spend per user | 1,102,653.33 |
| Time span | one year |

The average and median transaction counts are close, which suggests that the number of events per user is relatively balanced compared with many real implicit-feedback datasets. This is a deliberate design choice because the goal is to compare recommendation strategies rather than stress-test cold-start behavior in the current project stage.

### 3.8 Category-Level Distributions
The category distribution shows that the data are neither uniform nor dominated by a single spending domain. The top categories by transaction count are transport, restaurants, groceries, entertainment, and education. These results are plausible because those are categories with naturally repeated behavior. The high frequency of transport and restaurants reflects routine daily usage. Education appears strongly because the synthetic setting deliberately includes an educational ecosystem component, which is consistent with the project theme.

The top twelve categories by transaction count are:

| Category | Transactions |
|---|---:|
| transport | 10,672 |
| restaurants | 10,584 |
| groceries | 9,849 |
| entertainment | 9,789 |
| education | 9,222 |
| investments | 7,771 |
| electronics | 7,646 |
| insurance | 7,487 |
| money_transfer | 7,235 |
| utilities | 6,980 |
| home | 6,115 |
| healthcare | 6,048 |

This distribution matters for model design. Categories with many repeated events produce strong user-profile signals. Categories with fewer but semantically concentrated events may be more dependent on recency or textual representation. In a real banking environment, this distinction would influence whether one prioritizes lifetime aggregate features or recent-intent features.

[[FIGURE: Distribution of transaction counts across the main categories]]

### 3.9 Amount Distribution and Behavioral Heterogeneity
Transaction amounts in the synthetic data show heavy-tailed behavior, which is desirable because real financial activity is rarely normally distributed. Routine categories such as transport tend to have lower average amounts. Education or investment-linked operations have fewer but larger payments. This heterogeneity is important because the recommendation system should not treat all events as equally informative. A repeated low-value transaction in transport may indicate a strong stable routine, while an occasional large education payment may signal a high-intent but temporary phase.

This observation directly motivated the feature design used in the thesis. User profiles are not built only from counts; they combine normalized spend share and frequency share. Without this combination, the model would overemphasize either frequent microtransactions or rare high-value operations. The balance between the two becomes especially important in the time-decay model, where recent large transactions and stable repeated behavior compete as signals.

[[FIGURE: Distribution of total spend and median values across segments]]

### 3.10 Temporal Coverage
The dataset spans one year, which is sufficient to study recency and short-term windows without making the benchmark too sparse. Temporal coverage is essential because the advanced model relies on the difference between long-term and short-term user behavior. If the synthetic data covered only a short interval, the time-decay extension would be much less meaningful.

### 3.11 EDA Implications for Modeling
The exploratory analysis led to several modeling conclusions:
1. profile-based modeling is justified because category structure is strong and interpretable;
2. recency-aware modeling is justified because the data contain time information and potentially shifting preferences;
3. semantic offer representation is justified because the catalog mixes multiple product types with textual descriptions;
4. a pure collaborative filtering approach may be disadvantaged because the item catalog is still relatively small.

These conclusions shaped the experimental design. The project therefore evaluates a simple profile baseline, an implicit matrix factorization baseline, a neural collaborative filtering baseline, a hybrid semantic method, and a recency-aware advanced model. This model set is not arbitrary; it is directly aligned with the properties revealed by EDA while also including a nonlinear ML-oriented reference from the recommendation literature.

## 4. Problem Formulation and Evaluation Protocol
### 4.1 Ranking Task Formulation
Let `U` denote the set of users and `I` denote the set of candidate offers. For each user `u` in `U`, the system observes a transaction history consisting of timestamped category-labeled monetary events. From this history, the system constructs a user representation. Each offer `i` in `I` is represented by structured category targets and textual metadata. The goal is to learn a scoring function `s(u, i)` that allows offers to be ranked for each user. The recommendation output is a top-K list.

This formulation matches the practical business scenario. A banking application or notification system does not need a full calibrated probability distribution over all products. It needs a short list of plausible next offers that can be displayed or messaged to the client.

### 4.2 Why Top-K Ranking Is the Right Objective
In many real systems the recommendation space is constrained by interface and business logic. A mobile banking application typically shows only a small number of banners, cards, or product suggestions on the main screen. Therefore, ranking quality at the very top of the list matters much more than average score quality across all items. A model that slightly improves ranks 1 to 5 may have much more practical value than a model that improves average error but not top positions.

This is why the thesis emphasizes `Precision@5`, `Recall@5`, `MAP@5`, and `NDCG@5`. `Precision@5` measures how many of the top five recommended offers are relevant. `Recall@5` measures how well the list covers relevant items. `MAP@5` evaluates rank-sensitive average precision across the top positions. `NDCG@5` emphasizes whether the most relevant items appear earlier in the ranking.

### 4.3 Holdout Strategy
The evaluation protocol uses a leave-last-positive style holdout. For each user, the last positive interaction is kept as the target event, while earlier information is used for model construction. This protocol has several advantages. First, it prevents leakage from future interactions. Second, it is conceptually aligned with the idea of predicting the next relevant product exposure. Third, it remains simple enough to explain and reproduce.

### 4.4 Candidate Set and Seen-Item Exclusion
Recommendations are generated over the offer catalog while excluding already observed positive offers when appropriate. This reflects a practical assumption: once a user has already positively interacted with or converted on a given offer, the system may want to recommend something new rather than simply repeat the same product. In the present synthetic setup, this exclusion also makes the ranking task less trivial.

### 4.5 Metric Definitions and Interpretation
The four primary metrics are interpreted as follows. `Precision@5` answers the question: out of the first five recommended offers, what fraction is relevant? `Recall@5` answers how much of the relevant signal is captured in the first five positions. `MAP@5` is valuable because it is sensitive to the position of relevant offers inside the top-K list. `NDCG@5` is a standard top-rank quality metric that discounts lower positions. Together, these metrics provide complementary views of recommendation quality. The thesis does not rely on a single metric because model differences can appear in order-sensitive measures even when `Precision@5` is unchanged.

### 4.6 Baselines and Comparison Logic
The evaluation is organized around comparative benchmarking. The simplest interpretable baseline is profile similarity. The most classical interaction-based contrast is implicit matrix factorization. A nonlinear literature-grounded ML baseline is neural collaborative filtering. The most natural hybrid extension is profile plus semantic representation. The moderately advanced temporal extension is time-decay weighting. This comparison set is small enough to stay focused and large enough to support a meaningful discussion of trade-offs.

### 4.7 Threats to Evaluation Validity
Several limitations of the protocol must be acknowledged. First, the synthetic interaction generator is category-driven, which may favor category-profile methods. Second, the offer catalog contains only fifteen products, which compresses the ranking difficulty. Third, using one last positive event as holdout is a practical but simplified protocol; a real deployment would require broader temporal validation and potentially online A/B testing.

## 5. Methods and Implementation
### 5.1 End-to-End Pipeline Overview
The project was built as a reproducible pipeline rather than a single experimental notebook. This design choice matters because reproducibility is one of the core goals of the thesis. The pipeline includes data generation, preprocessing, exploratory analysis, model training or scoring, metric calculation, robustness analysis, and service-level inference. By structuring the work as a modular pipeline, the project supports repeated execution and easy extension.

At a high level, the workflow is:
1. generate users, transactions, offers, and interactions;
2. construct train/evaluation splits;
3. compute model-specific user and offer representations;
4. generate top-K recommendations;
5. evaluate ranking quality;
6. export reports, figures, and tables;
7. serve recommendations through an API prototype.

This structure also has pedagogical value for the thesis. It demonstrates not only model comparison, but the engineering discipline required to move from idea to reproducible artifact.

[[FIGURE: Overall architecture of the personalization pipeline]]

### 5.2 Feature Engineering Philosophy
Feature engineering in the project is driven by interpretability and alignment with domain logic. Rather than immediately use a black-box model, the thesis first constructs behavior summaries that are understandable in banking terms. Two central signals are used:
1. spend share by category;
2. frequency share by category.

Spend share captures financial intensity. Frequency share captures routine prevalence. A user who makes many small transport transactions and a few large investment transactions may look very different depending on which of these signals is emphasized. Combining them makes the representation more balanced.

In addition, user behavior can be transformed into a lightweight text summary for semantic matching. Offer descriptions are represented with TF-IDF, which is computationally cheap and transparent. This choice is appropriate for a thesis baseline because it allows the semantic layer to be analyzed without the confounding cost of large pretrained models.

### 5.3 Profile Similarity Baseline
The profile similarity model is the most interpretable baseline in the project. For each user, the model aggregates transactions into a category vector. The vector combines normalized spend share and normalized frequency share with fixed weights. For each offer, the model constructs a target-intent vector from the offer's linked categories. The final recommendation score is a similarity between the user profile and the offer profile.

The strength of this baseline is that it directly expresses a plausible business heuristic: users should receive offers aligned with the categories in which they spend money or act frequently. The model is easy to explain to a supervisor, a nontechnical stakeholder, or a banking product manager. It also provides a meaningful reference for more complex approaches. If a sophisticated method cannot beat such a baseline, the added complexity may not be justified.

Another important advantage is robustness to sparse user-offer interaction data. Because the model uses transaction behavior rather than only observed conversions, it remains informative even when direct interaction logs are limited.

[[FIGURE: User-profile construction and final offer scoring scheme]]

### 5.4 Implicit Matrix Factorization Baseline
The implicit MF baseline treats user-offer interactions as the primary signal. The interaction matrix is factorized into lower-dimensional latent spaces using `TruncatedSVD`. Recommendation scores are reconstructed from latent user and item representations.

This baseline represents the collaborative-filtering perspective. It is included because collaborative filtering is a standard recommendation strategy and because it can sometimes discover useful structure that is not obvious from explicit metadata. For example, two offers may appear semantically unrelated but still attract similar users through latent behavior patterns.

However, the performance of this model depends strongly on the richness of the interaction structure. In the current project, interactions are generated from behavioral compatibility and the item set is small. As a result, the latent structure may not contain enough information to outperform profile-based models. The empirical results confirm this concern: implicit MF is consistently weaker than the best alternative approaches in this synthetic setup.

### 5.5 Neural Collaborative Filtering Baseline
Neural collaborative filtering (NCF) was added as a compact nonlinear ML baseline motivated by the literature review. The model uses trainable user and offer embeddings followed by a small multilayer scoring block. In conceptual terms, this gives the benchmark a representative neural recommender without requiring the full engineering weight of a large sequential architecture.

The role of NCF in this thesis is methodological rather than promotional. It provides a more serious machine learning reference point than purely linear or heuristic baselines and helps answer an important question: does simply moving to a nonlinear neural architecture improve recommendation quality in the current data regime? In the present project, the answer is only partially. NCF is stronger than simple implicit MF on some synthetic ranking metrics, but it still trails the best profile-based approaches.

This is an informative result. It suggests that model class alone is not enough; richer signal structure, larger catalogs, and more realistic sequential dynamics are also needed if neural recommendation is expected to dominate simpler baselines.

### 5.6 Hybrid Semantic Model
The hybrid semantic model combines profile similarity with text-based similarity. User interests are transformed into a text-like summary derived from behavioral categories. Offer descriptions are represented through TF-IDF vectors. The semantic similarity between the user summary and the offer description is then combined with the structured category-profile score.

This model tests whether lightweight NLP can improve recommendation without moving to heavy transformer architectures. The idea is important because the initial project concept explicitly considered semantic similarity between transaction themes and educational courses. After the thesis topic broadened from educational offers to a more general banking offer catalog, the semantic component remained relevant as a bridge between behavioral categories and product descriptions.

The hybrid model is also methodologically attractive because it represents a middle ground between rules and representation learning. It can capture lexical overlap and thematic affinity without sacrificing transparency.

### 5.7 Time-Decay Advanced Model
The time-decay model is the main advanced approach implemented in the thesis. Its motivation is straightforward: recent transactions may better reflect current needs than older transactions. Rather than replace the whole profile design, the model extends it by assigning exponentially decaying weights to transactions based on recency. It then blends long-term and short-term user representations.

The model contains several tunable components:
1. `decay_rate`, which controls how fast old transactions lose importance;
2. `short_term_days`, which defines the window of recent behavior;
3. `short_term_weight`, which determines how much the recent profile contributes relative to the lifetime profile;
4. `spend_weight` and `freq_weight`, which control the balance between monetary and count-based signals.

This design is attractive because it adds temporal intelligence while remaining interpretable. Unlike a Transformer, the model does not require complex training infrastructure. Yet it still tests a meaningful research hypothesis: whether dynamic emphasis on recent spending improves top-rank quality.

### 5.8 Hyperparameter Search
The time-decay model was tuned through a grid search over 108 configurations. The search explored several decay rates, short-term windows, and blend weights. The best configuration in the main reference run was:

| Parameter | Best Value |
|---|---:|
| decay_rate | 0.01 |
| short_term_days | 30 |
| short_term_weight | 0.20 |
| spend_weight | 0.60 |
| freq_weight | 0.40 |

This result suggests that the strongest model in the current setting does not rely on extreme recency. Instead, it favors a moderate short-term contribution on top of a still dominant long-term profile. This is intuitively plausible. Banking behavior tends to contain both stable routine patterns and short-term need shifts. Overweighting the recent window too much would erase useful long-term structure.

### 5.9 Engineering Structure and Reproducibility
The codebase is organized into `src/data`, `src/models`, `src/evaluation`, `src/pipelines`, and `src/service`. This separation is not only stylistic. It ensures that the data generation logic, model logic, evaluation logic, and service logic are cleanly separated. Such modularization improves maintainability and makes the thesis easier to extend after the consultation stage.

Multiple pipeline scripts are provided:
- a baseline pipeline for the initial full run;
- an EDA report generator;
- a neural collaborative filtering baseline pipeline;
- a LightGCN baseline pipeline;
- a time-decay tuning pipeline;
- a robustness and segment-analysis pipeline;
- a multi-seed benchmark pipeline;
- a minimal real-data validation pipeline on a retail transaction dataset;
- a SASRec real-data validation pipeline;
- a service demo output generator;
- a Word thesis builder.

The presence of these scripts strengthens the methodological value of the thesis. The project is not a single notebook with fragile state; it is a small reproducible research environment. The source code and generated artifacts are included in the electronic thesis package, which makes the project auditable as a software deliverable rather than only as prose.

[[FIGURE: Project structure and the main system modules]]

### 5.10 Why More Complex Models Were Not Prioritized
An important implementation decision was not to make the heaviest graph or sequential neural models the central serving story of the thesis. This decision was not due to ignorance of the relevant literature. Rather, it was a scoped engineering choice. Under a tight thesis schedule, the project prioritizes:
1. a strong interpretable baseline;
2. at least one meaningful advanced extension and one nonlinear ML baseline;
3. robustness analysis;
4. practical deliverables such as an API prototype and Word draft.

Within that scope, the project still implements both LightGCN and a compact SASRec branch, but treats them as auxiliary advanced validations rather than as the final production recommendation choice. This is the right trade-off for the current thesis stage. A weakly validated complex model would add technical glamour but not necessarily scientific value. In contrast, a smaller but reproducible benchmark with clear analysis is more defensible before a supervisor and a committee.

## 6. Experiments and Results
### 6.1 Experimental Setup
All experiments were run on the synthetic dataset described earlier. The main reference run uses 800 users, 112,626 transactions, 15 offers, and 6,442 user-offer interactions. Evaluation was conducted with top-K ranking at `K = 5`. For each user, the last positive interaction served as the holdout target.

The models compared in the main benchmark are:
1. `Profile Similarity`;
2. `Implicit MF`;
3. `Neural CF`;
4. `Hybrid Semantic`;
5. `Time-Decay`.

This lineup allows the study to compare interpretable structured matching, latent collaborative filtering, nonlinear neural recommendation, hybrid text-behavior matching, and temporal weighting within one protocol.

### 6.2 Main Single-Run Results
The main single-run experiment produced the following results:

| Model | Precision@5 | Recall@5 | MAP@5 | NDCG@5 |
|---|---:|---:|---:|---:|
| Profile Similarity | 0.0923 | 0.4613 | 0.2349 | 0.2906 |
| Implicit MF | 0.0660 | 0.3300 | 0.1460 | 0.1911 |
| Neural CF | 0.0678 | 0.3387 | 0.1518 | 0.1975 |
| Hybrid Semantic | 0.0918 | 0.4587 | 0.2334 | 0.2888 |
| Time-Decay | 0.0923 | 0.4613 | 0.2387 | 0.2936 |

The most immediate observation is that the time-decay model achieves the best `MAP@5` and `NDCG@5`, while `Precision@5` and `Recall@5` are essentially tied with the strongest baseline. This means that the value of time-decay lies mostly in ordering relevant offers better, not in finding a dramatically different set of top-five candidates.

The second observation is the relatively weak performance of implicit MF and Neural CF compared with the strongest profile-based methods. This result is important because it rejects a naive assumption that latent-factor or neural recommenders automatically outperform structured behavior models. In the current synthetic setting, the collaborative signal is weaker than the structured behavioral signal.

At the same time, Neural CF remains useful analytically. It serves as a nonlinear ML baseline grounded in the recommendation literature. On the current synthetic dataset, it slightly improves over implicit MF in `MAP@5` and `NDCG@5`, but still remains well below `Profile Similarity`, `Hybrid Semantic`, and `Time-Decay`. This suggests that moving to a neural architecture alone does not guarantee gains without richer histories, a larger item catalog, or a stronger sequential signal.

[[FIGURE: Comparison of NDCG@5 and MAP@5 across the five implemented models]]

### 6.3 Interpretation of Single-Run Results
The single-run benchmark supports several interpretations.

First, transaction-derived category behavior is informative enough to support meaningful recommendation quality. The profile baseline already reaches nontrivial `MAP@5` and `NDCG@5`, which means that even an interpretable category-matching approach can retrieve useful offers.

Second, temporal weighting helps, but only modestly. The magnitude of improvement is small, yet it appears exactly where expected: in metrics that reward better ordering at the top ranks. This is consistent with the hypothesis that recency changes ranking detail more than broad relevance coverage.

Third, the hybrid semantic model remains very competitive. On the single reference run it is slightly below time-decay, but the gap is small enough that a richer semantic layer could plausibly become stronger under a larger or more varied catalog.

Fourth, the addition of Neural CF improves the scientific positioning of the benchmark even though it does not become the top performer. The thesis can therefore compare not only interpretable and temporal baselines, but also a representative nonlinear recommender from the modern literature.

### 6.4 Time-Decay Tuning Results
The grid search over 108 configurations provides additional insight. The best-performing configuration uses `decay_rate = 0.01`, `short_term_days = 30`, `short_term_weight = 0.20`, `spend_weight = 0.60`, and `freq_weight = 0.40`. Several nearby configurations perform almost as well, which suggests that the model is not extremely fragile.

The best few configurations all point to the same broader conclusion: moderate recency emphasis works better than very aggressive short-term weighting. This is an intuitively valuable result for a banking domain. Clients may have changing needs, but their long-run transaction identity still carries substantial information. A recommender that forgets history too quickly risks becoming unstable.

[[FIGURE: Time-decay hyperparameter search results]]

### 6.5 Segment-Level Analysis
To understand whether the benefits of time-aware modeling depend on user type, the thesis evaluates per-segment metrics. Segment-level analysis shows that the time-decay model is not uniformly stronger across the full population.

The following qualitative pattern was observed:
1. `family`, `investor`, and `student` segments benefit from time-decay relative to the static profile baseline;
2. `daily_life`, `digital_pro`, and `traveler` are slightly better served by the static profile baseline.

This finding is important because it connects model performance to behavioral regimes. For `family`, recent changes in household behavior may carry meaningful short-term signals, for example around health, home, or child-related expenses. For `investor`, recent movement between financial categories may be especially informative. For `student`, temporary phases such as enrollment, exam preparation, or mobility changes may matter. By contrast, `daily_life` behavior is relatively stable and may not require a strong recency adjustment.

Selected segment-level results for `NDCG@5` illustrate this:

| Segment | Profile Baseline | Time-Decay | Delta |
|---|---:|---:|---:|
| daily_life | 0.2913 | 0.2872 | -0.0040 |
| digital_pro | 0.2840 | 0.2787 | -0.0053 |
| family | 0.2790 | 0.2954 | +0.0164 |
| investor | 0.3282 | 0.3370 | +0.0088 |
| student | 0.2611 | 0.2785 | +0.0175 |
| traveler | 0.2998 | 0.2897 | -0.0101 |

This table strengthens the thesis discussion because it moves beyond average metrics and reveals user heterogeneity.

[[FIGURE: Segment-wise metrics across the main behavioral groups]]

### 6.6 Bootstrap Uncertainty Analysis
Point estimates alone are not enough to judge whether one model is convincingly better than another. To address this, the thesis applies bootstrap resampling at the user level. The comparison focuses on the difference between the time-decay model and the profile baseline because these two approaches are the closest on the main benchmark.

In the single-run bootstrap analysis with 3,000 resamples, the mean differences are positive:
- `NDCG@5` mean difference: `+0.003010`;
- `MAP@5` mean difference: `+0.003813`.

However, the 95% confidence intervals still include zero:
- `NDCG@5`: `[-0.010062, 0.015470]`;
- `MAP@5`: `[-0.007459, 0.015189]`.

This means the improvement is directional but not statistically definitive under the current setup. This is a valuable scientific result. Rather than overclaiming the superiority of the advanced model, the thesis demonstrates appropriate caution and uses uncertainty analysis to calibrate interpretation.

[[FIGURE: Bootstrap comparison of time-decay and the profile baseline]]

### 6.7 Multi-Seed Benchmark
Randomness in synthetic data generation can influence results. Therefore, the benchmark was repeated over five random seeds: 7, 13, 21, 42, and 77. The mean results are shown below:

| Model | Mean Precision@5 | Mean Recall@5 | Mean MAP@5 | Mean NDCG@5 |
|---|---:|---:|---:|---:|
| Hybrid Semantic | 0.0910 | 0.4550 | 0.2275 | 0.2833 |
| Time-Decay | 0.0916 | 0.4578 | 0.2254 | 0.2824 |
| Profile Similarity | 0.0903 | 0.4515 | 0.2262 | 0.2815 |
| Implicit MF | 0.0724 | 0.3618 | 0.1680 | 0.2155 |

This benchmark slightly changes the narrative. On the single reference run, the time-decay model is best. Across multiple seeds, the hybrid semantic model achieves the best mean `NDCG@5`, while the profile baseline and time-decay remain extremely close. Implicit MF remains clearly weaker.

This result highlights an important methodological lesson: in a synthetic setup with a small item catalog and strong category structure, the top non-MF models may differ only marginally. Therefore, reproducibility and stability become more important than celebrating a tiny point-estimate win.

[[FIGURE: Comparison of multi-seed benchmark results for the key models]]

### 6.8 Minimal Real-Data Validation on Online Retail
To reduce the risk that the full thesis remains purely synthetic, a minimal external validation was also conducted on the public `Online Retail` transaction dataset. After cleaning and filtering, the usable benchmark contained 2,992 users, 1,499 items, and 201,226 positive user-item interactions. This dataset does not belong to the banking domain, but it provides a real transaction log with enough structure to test whether the pipeline transfers beyond synthetic generation.

The real-data ranking results at `K = 10` showed a different ordering from the synthetic benchmark:

| Model | MAP@10 | NDCG@10 |
|---|---:|---:|
| Implicit MF | 0.1025 | 0.1296 |
| Item KNN | 0.0742 | 0.0952 |
| Neural CF | 0.0314 | 0.0434 |
| LightGCN | 0.0075 | 0.0114 |
| Popularity | 0.0165 | 0.0241 |

In a separate sequence-aware run on the same transaction log, SASRec reached `MAP@10 = 0.0063` and `NDCG@10 = 0.0092`. This confirms that the thesis includes a real sequential branch, but also shows that the current implementation still requires further tuning before it can compete with the strongest simpler baselines.

This outcome is important for interpretation. It shows that the pipeline is not tied exclusively to the synthetic banking-like generator and can operate on real purchase histories. At the same time, it does not prove banking-domain validity, because retail purchases and banking transactions differ substantially. Therefore, this experiment should be interpreted as a sanity check of external reproducibility rather than as final industrial evidence.

### 6.9 Why Implicit MF and Neural CF Are Weaker Here
The repeated underperformance of implicit MF and Neural CF on the synthetic benchmark deserves explicit interpretation. There are at least three plausible reasons. First, the offer catalog is small. Collaborative filtering often benefits from richer interaction topology and a larger item set. Second, the synthetic interaction generator is behaviorally structured around categories, which naturally favors content-aligned methods. Third, the interaction signal may not contain enough latent complexity to justify factorization or nonlinear neural scoring.

This does not mean collaborative filtering or neural recommenders are unimportant in banking recommendation overall. It means only that, under the present synthetic assumptions, they are not the strongest tools. The real-data retail validation partly supports this interpretation by showing that collaborative structure becomes more useful once the benchmark is based on actual purchase logs rather than synthetic category-driven interactions.

### 6.10 Representative Recommendation Cases
Qualitative inspection of sample recommendations helps interpret model behavior. For `U00001`, the service demo suggests `Balanced Finance Bundle`, `Auto and Mobility Protection`, `Mortgage Refinance Program`, `Health and Family Insurance Bundle`, and `Beginner Investment Account`. This recommendation list is coherent for a user whose behavior reflects finance, home, and mobility-linked spending.

For `U00002`, the top suggestions are `Auto and Mobility Protection`, `Travel Miles Premium Card`, `Premium Lifestyle Subscription`, `EdTech Learning Subscription`, and `Digital Security Package`. This looks like a more mobile or travel-adjacent profile enriched by digital services. Such qualitative examples show that the model outputs are interpretable and business-plausible, even when the benchmark is synthetic.

### 6.11 Summary of Experimental Findings
The experiments support six key findings:
1. transaction behavior is informative enough for useful top-K offer ranking;
2. the profile baseline is strong and difficult to beat decisively;
3. recency-aware modeling improves ordering quality in the reference run and remains competitive overall;
4. lightweight semantics are competitive and become the top mean `NDCG@5` model in the multi-seed benchmark;
5. the nonlinear Neural CF baseline improves the benchmark design but does not outperform the strongest interpretable models on the synthetic setup;
6. minimal real-data validation shows that the pipeline transfers to real transaction logs, where collaborative structure becomes relatively more useful.

## 7. Service Prototype and Deployment Considerations
### 7.1 Purpose of the Service Prototype
The project is not limited to offline experiments. To demonstrate practical applicability, a minimal service prototype was implemented using FastAPI. The purpose of this service is not to simulate a full industrial recommendation platform, but to show that the developed models can be wrapped into an inference interface and used to return personalized offer lists for a given user identifier.

The presence of a service prototype strengthens the project in two ways. First, it shows that the work is not purely theoretical. Second, it helps bridge the gap between thesis artifacts and real product implementation concerns.

### 7.2 API Design
The service currently exposes two endpoints:
1. GET /health for service health checking;
2. GET /recommend/{user_id}?top_k=5 for personalized recommendation retrieval.

This interface is intentionally simple. It is sufficient for a prototype, easy to test during a consultation, and easy to explain in the thesis. The recommendation endpoint returns offer identifiers, product metadata, ranking position, and model score. This is enough to support demo scenarios, manual inspection, and future UI integration.

[[FIGURE: Swagger UI screenshot of the recommendation API]]

### 7.3 Model Behind the Service
The current API uses the tuned time-decay model as the serving model. This choice is pragmatic. Even though multi-seed results show that top models are very close, the time-decay variant is a reasonable serving candidate because it performed best on the reference run and represents the main advanced contribution of the thesis.

From an engineering perspective, the service layer is decoupled from the training pipeline. This means that replacing the serving model with a hybrid semantic or future sequential model would be relatively straightforward.

### 7.4 Inference Flow
At inference time, the service performs the following steps:
1. loads the required synthetic data and prepared model artifacts;
2. constructs or retrieves the user representation;
3. scores candidate offers;
4. ranks them;
5. returns the top-K list with metadata.

In a production system, one would additionally need caching, model versioning, logging, rate limiting, authentication, and potentially candidate generation at scale. The thesis does not implement these features fully, but it explicitly identifies them as future engineering requirements.

[[FIGURE: Example API response with top personalized banking offers]]

### 7.5 Monitoring and Retraining Considerations
Even a simple banking recommender requires monitoring. The service should track:
- input data freshness;
- recommendation coverage across offers;
- score distribution drift;
- latency and error rate;
- conversion or engagement proxies if online data exist.

Retraining policy is also important. A static model can become stale if user behavior changes or the offer catalog evolves. In a real bank, retraining cadence would depend on data availability, campaign dynamics, and system architecture. The current project does not implement automated retraining, but the modular pipeline makes such an extension natural.

### 7.6 Privacy and Responsible Deployment
Any real banking deployment would require strict privacy controls, access management, and compliance review. A key advantage of the current synthetic prototype is that it allows architectural experimentation without touching sensitive client data. This can be viewed as a safe pre-production stage: the bank can validate the pipeline logic, interfaces, and monitoring patterns before integrating actual restricted data.

### 7.7 Value of the Service Prototype for the Thesis
The service prototype adds practical depth to the thesis. It demonstrates that recommendation research is not only about comparing metrics in notebooks, but also about exposing results through a usable interface. For a master thesis, this is a strong sign of project completeness.

## 8. Discussion
### 8.1 Main Interpretation
The central conclusion of the thesis is not that one model vastly dominates all others. Rather, the main achievement is the construction of a reproducible and analytically grounded personalization pipeline for a banking-inspired environment. The strong result is methodological rather than sensational: multiple reasonable recommenders were compared under a transparent protocol, and the conclusions were stress-tested with uncertainty and stability analysis.

This conclusion is valuable because real-world machine learning often suffers from a mismatch between ambitious model claims and fragile experimental setups. In contrast, the current thesis favors careful scope management and defensible evidence.

### 8.2 Why the Gains Are Modest
The performance differences among the best non-MF models are modest for understandable reasons. The data generator is category-centered, the offer catalog is small, and the user behavior segments are fairly interpretable. Under these conditions, a profile baseline already captures much of the available signal. Advanced models can refine the ordering, but they do not transform the problem into something radically different.

This is not a failure. In many industrial domains, the best-performing model is only slightly better than a strong baseline. What matters is whether the improvement is stable, explainable, and operationally worth the complexity. The current thesis demonstrates exactly this kind of thinking.

### 8.3 Practical Implications for Banking Personalization
Several practical insights follow from the experiments.

First, transaction-derived category profiles can already support a useful recommendation layer. Banks that lack a mature large-scale recommender stack may still start from interpretable profile matching.

Second, recency matters, but not uniformly. If a bank wants to personalize offers for rapidly changing user segments, time-aware weighting is a sensible improvement. If the user base is dominated by stable daily-behavior patterns, a static profile model may be sufficient as an initial production baseline.

Third, semantic product representation is promising. Even a lightweight text model remains competitive. This suggests that richer offer descriptions and more expressive embeddings may become increasingly valuable as the product catalog grows.

Fourth, external validation matters. Even a small real-data experiment can change the relative ranking of models and prevent overconfident conclusions based only on synthetic benchmarks.

### 8.4 Limitations
The thesis has several important limitations. The first limitation is external validity. The main benchmark is synthetic. This means the results cannot be directly interpreted as expected live banking conversion gains. The second limitation is catalog scale. With only fifteen offers, the candidate space is relatively small. A larger and more dynamic catalog would likely make semantic modeling and collaborative structure more important. The third limitation is evaluation protocol simplicity. The leave-last-positive holdout is reasonable, but real deployment would require temporal backtesting, campaign-aware validation, and eventually online experimentation. The fourth limitation is that the real-data validation was performed on a retail rather than banking dataset, so it should be understood only as a transferability check. The fifth limitation is model scope. The project does not yet include a full sequential Transformer or graph recommender. This was a conscious scope decision rather than a conceptual omission.

### 8.5 Ethical, Privacy, and Fairness Considerations
Personalization in banking is not only a technical problem. It also raises ethical and privacy questions. Recommendations can affect financial decision-making, product exposure, and customer behavior. Therefore, a responsible system should avoid manipulative targeting, monitor bias across user groups, and ensure that the model does not expose sensitive inferred traits inappropriately.

The synthetic setup of this thesis cannot solve these issues completely, but it is a responsible first step. Because no real personal data are used, the project avoids direct privacy risk while still allowing the researcher to think through the architecture and evaluation framework.

### 8.6 Future Work
The most promising directions for future work are:
1. richer synthetic generation with explicit seasonality, life-event shifts, and merchant-level structure;
2. larger offer catalogs with more nuanced textual descriptions;
3. sequential recommendation models such as SASRec or BERT4Rec;
4. graph-based modeling of users, offers, merchants, and categories;
5. counterfactual or uplift-oriented evaluation more aligned with campaign response;
6. migration from synthetic to privacy-preserving real data in collaboration with an industrial partner.

These directions show that the thesis is not a closed endpoint. It is a foundation.

## 9. Conclusion
This thesis studied personalization of banking offers from user transaction activity in a reproducible synthetic environment complemented by a minimal real-data validation experiment. The work combined literature review, synthetic dataset design, exploratory data analysis, model comparison, robustness analysis, external validation on a public retail transaction log, and service prototyping.

The project showed that transaction-driven profiles are informative enough to support nontrivial top-K recommendation quality. Among the compared models, profile similarity, hybrid semantic ranking, and time-decay weighting all performed competitively on the synthetic benchmark, while implicit matrix factorization and Neural CF were weaker there. The time-decay model produced the strongest reference-run ordering metrics, but multi-seed results demonstrated that the best non-MF models remain close. The additional Online Retail validation showed that the pipeline also transfers to real transaction logs and that collaborative structure can become more useful outside the synthetic setting. This makes careful interpretation and reproducibility more important than claiming a dramatic universal winner.

The thesis therefore contributes not only a set of metrics, but a coherent methodological artifact: an end-to-end pipeline for studying banking personalization when real financial data are unavailable, plus a first sanity check on real transactional data. This outcome is valuable both as a master project result and as a foundation for further research with richer data and stronger models.

## References
1. Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua. Neural Collaborative Filtering. 2017. URL: https://arxiv.org/abs/1708.05031 (accessed 14.03.2026).
2. Wang-Cheng Kang, Julian McAuley. Self-Attentive Sequential Recommendation. 2018. URL: https://arxiv.org/abs/1808.09781 (accessed 14.03.2026).
3. Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, Peng Jiang. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. 2019. URL: https://arxiv.org/abs/1904.06690 (accessed 14.03.2026).
4. Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. 2020. URL: https://arxiv.org/abs/2002.02126 (accessed 14.03.2026).
5. Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, et al. Wide & Deep Learning for Recommender Systems. 2016. URL: https://arxiv.org/abs/1606.07792 (accessed 14.03.2026).
6. Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. 2017. URL: https://arxiv.org/abs/1703.04247 (accessed 14.03.2026).
7. Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, Guangzhong Sun. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. 2018. URL: https://arxiv.org/abs/1803.05170 (accessed 14.03.2026).
8. Nils Reimers, Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. 2019. URL: https://arxiv.org/abs/1908.10084 (accessed 14.03.2026).
9. Lifeng Yi, et al. Pcard: Personalized Restaurants Recommendation from Card Transactions. WWW. 2019. URL: https://users.cs.utah.edu/~lifeifei/papers/pcard-www19.pdf (accessed 14.03.2026).
10. Sayan Biswas, et al. A link prediction-based recommendation system using transactional data. Scientific Reports. 2023. URL: https://www.nature.com/articles/s41598-023-34055-5 (accessed 14.03.2026).
11. Jungim Baek, et al. Merchant Recommender System Using Credit Card Payment Data. Electronics. 2023. URL: https://www.mdpi.com/2079-9292/12/4/811 (accessed 14.03.2026).
12. Adaptive CF with Personalized Time Decay for Financial Product Recommendation. 2023. URL: https://arxiv.org/abs/2308.01208 (accessed 14.03.2026).
13. Marc Boulle, et al. Synthesizing Credit Card Transactions. 2019. URL: https://arxiv.org/abs/1910.03033 (accessed 14.03.2026).

## Appendix A. Synthetic Segment Logic and Category Mapping
The following appendix provides a more operational view of how synthetic behavior segments map to dominant transaction categories. The goal is not to claim sociological realism, but to document the assumptions used by the data generator.

| Segment | Primary Categories | Secondary Categories | Typical Recommendation Themes |
|---|---|---|---|
| daily_life | groceries, transport, utilities | cash_withdrawal, restaurants | cashback cards, autopay products, savings |
| digital_pro | electronics, money_transfer, entertainment | education, subscriptions | digital security, premium digital services |
| family | groceries, home, healthcare | insurance, utilities | family insurance, deposits, home-related products |
| investor | investments, money_transfer, education | utilities, insurance | investment account, financial bundle, security service |
| student | education, transport, entertainment | restaurants, electronics | student card, learning subscription, budget-friendly offers |
| traveler | travel, transport, restaurants | insurance, entertainment | travel card, mobility insurance, lifestyle subscription |

These mappings were used only as generative priors. The final transactions remain stochastic, so users inside the same segment still vary individually.

An important design principle was to ensure overlap between segments. For example, both traveler and daily_life may have strong transport spending, but for different overall profiles. This overlap prevents the classification of users into offers from becoming too deterministic and makes the recommendation problem more realistic.

## Appendix B. Extended Offer Notes
This appendix describes the role of each synthetic offer in the benchmark.

| Offer ID | Offer Name | Role in Benchmark |
|---|---|---|
| O001 | Smart Daily Cashback Card | a broad baseline banking offer relevant to routine spending users |
| O002 | Travel Miles Premium Card | mobility-focused premium card |
| O003 | Beginner Investment Account | entry-level finance product for users with investment-like behavior |
| O004 | Flexible Personal Loan | classic credit product for lifestyle or home-related needs |
| O005 | Mortgage Refinance Program | longer-horizon household finance offer |
| O006 | Health and Family Insurance Bundle | protection-oriented product with family relevance |
| O007 | Auto and Mobility Protection | mobility insurance for transport and travel users |
| O008 | Family Savings Deposit | conservative savings product with household orientation |
| O009 | EdTech Learning Subscription | ecosystem-style partner offer aligned with education |
| O010 | Premium Lifestyle Subscription | entertainment and dining focused partner subscription |
| O011 | Utility AutoPay Cashback | card-like utility automation offer |
| O012 | Digital Security Package | service offer aligned with digital and financial safety |
| O013 | Student Smart Start Card | youth-focused entry card |
| O014 | Cash Management Plus | utility-like service around cash flow convenience |
| O015 | Balanced Finance Bundle | broad cross-sell bundle spanning investment and protection |

The catalog intentionally mixes narrowly focused and broad offers. Narrow offers help reveal user specificity, while broader bundles act as generalist recommendations. In a real bank, such a mix is common because some products are niche and others are cross-segment.

## Appendix C. Hyperparameter Search Details
The time-decay sweep explored combinations of decay intensity, short-term window size, profile blending weight, and spend-frequency balance. The goal was not exhaustive global optimization, but a structured search over plausible values.

| Parameter | Tried Values |
|---|---|
| decay_rate | 0.005, 0.01, 0.02 |
| short_term_days | 30, 60, 90 |
| short_term_weight | 0.20, 0.35, 0.50 |
| spend_weight | 0.60, 0.70, 0.80 |
| freq_weight | 0.40, 0.30, 0.20 |

The top configurations cluster around low-to-moderate decay and a relatively small recent-behavior weight. This is evidence that the benchmark rewards temporal sensitivity without requiring the model to discard long-term preference structure.

It is also noteworthy that multiple near-best configurations remain close in NDCG@5. This suggests that the advanced model is not only competitive but also reasonably stable around the best settings.

## Appendix D. Reproducibility and Project Structure
The project was organized to support regeneration of data and results from scratch. The main folders are:

| Path | Purpose |
|---|---|
| data/ | synthetic inputs and derived datasets |
| docs/ | thesis drafts, checkpoint materials, presentation texts |
| 
otebooks/ | executed EDA notebooks |
| eports/ | metrics, figures, robustness outputs |
| scripts/ | PowerShell and Python entrypoints |
| src/data/ | data generation logic |
| src/models/ | recommender implementations |
| src/evaluation/ | ranking metrics |
| src/pipelines/ | experiment orchestration |
| src/service/ | API prototype |

The project can be reproduced in the following order:
1. install dependencies from equirements-dev.txt;
2. generate data and baseline results;
3. run EDA reporting;
4. run advanced model search;
5. run robustness analysis and multi-seed benchmark;
6. build the thesis Word document.

This type of reproducibility is especially important in an academic project where multiple checkpoints are evaluated over time. Each milestone should leave a traceable artifact rather than only verbal progress claims.

## Appendix E. Example Recommendation Scenarios
This appendix provides qualitative scenarios for several sample users. The examples are not intended as proof of business value, but as interpretive support for the ranking outputs.

### Scenario 1: Finance- and home-oriented user
For U00001, the service returns Balanced Finance Bundle, Auto and Mobility Protection, Mortgage Refinance Program, Health and Family Insurance Bundle, and Beginner Investment Account. This list reflects a profile where finance-related and household-related categories both matter. The ranking mixes protection, finance, and longer-horizon household products, which is reasonable for a user whose activity is not confined to one narrow category.

### Scenario 2: Travel and mobility user
For U00002, the model surfaces Auto and Mobility Protection and Travel Miles Premium Card at the top, followed by a lifestyle subscription and digital-service offers. This ordering is intuitively coherent because transport and travel behavior frequently co-occur with insurance and rewards products in banking ecosystems.

### Scenario 3: Family-oriented user
A representative family user tends to receive combinations of Health and Family Insurance Bundle, Family Savings Deposit, Mortgage Refinance Program, and Utility AutoPay Cashback. These products align with routine household spending, home-related concerns, and financial stability goals.

### Scenario 4: Student-oriented user
A representative student user is likely to receive Student Smart Start Card, EdTech Learning Subscription, Premium Lifestyle Subscription, and occasionally a beginner financial product such as Beginner Investment Account. This combination reflects the idea that student behavior is not only budget-sensitive but also associated with education and early financial onboarding.

### Scenario 5: Investor-oriented user
An investor user often receives Beginner Investment Account, Balanced Finance Bundle, and Digital Security Package. This is a useful example because it shows how finance-related behavior can support recommendations that are not identical but complementary: investment access, product bundling, and security-oriented services.

The value of these scenarios is interpretability. A ranking model in banking should ideally produce outputs that a human stakeholder can understand and defend.

## Appendix F. Extended Literature Notes
This appendix provides concise thesis-oriented notes on the most relevant references used in the study.

### F.1 Neural Collaborative Filtering
Neural Collaborative Filtering [1] is important because it reframes latent user-item interaction as a nonlinear matching problem. For the current thesis, its main value is methodological. It establishes a strong reference point for implicit-feedback recommendation and motivates the inclusion of a collaborative baseline, even though the implementation here remains simpler than the original neural architecture.

### F.2 SASRec
SASRec [2] is one of the most relevant sequential references for the project. Its attention-based modeling is a natural candidate for transaction histories because it can learn which recent actions matter most. The present thesis includes a compact SASRec branch on the real transaction log. In the current configuration it underperforms the strongest simpler baselines, but its implementation is still valuable because it turns the sequence-aware direction into an executed experiment rather than only a future-work claim.

### F.3 BERT4Rec
BERT4Rec [3] extends sequential recommendation by learning bidirectional contextual dependencies. This is especially relevant in domains where behavior patterns are multi-step and contextual rather than strictly next-item local. In a future version of the banking project, such a model could capture patterns like transitions from education spending to savings interest, or from home spending to refinancing offers.

### F.4 LightGCN
LightGCN [4] is central to modern graph recommendation because it simplifies graph convolution while preserving strong performance. The thesis uses this reference both as conceptual motivation and as an implemented graph baseline. In the current experiments, the LightGCN branch does not surpass the strongest profile-based or MF baselines, but it demonstrates that graph recommendation can be integrated into the same evaluation stack and compared honestly under the same reproducibility standards.

### F.5 Wide & Deep
Wide & Deep [5] is a foundational industrial recommendation paper because it combines memorization and generalization. In the context of this thesis, the paper is useful for thinking about how a banking recommender could combine explicit business rules or feature crosses with learned distributed representations.

### F.6 DeepFM
DeepFM [6] is particularly relevant when the recommendation problem can be expressed as rich tabular feature interaction. Banking personalization fits this template well: segment features, category shares, transaction statistics, and offer types can all be encoded structurally. The current thesis stops at a lighter model family, but DeepFM remains one of the most plausible next-step architectures once the dataset becomes richer.

### F.7 xDeepFM
xDeepFM [7] extends feature interaction modeling further by combining explicit and implicit feature interactions. It is cited in the thesis because it reflects the broader industrial lesson that recommendation quality often improves not because of one single signal, but because different interaction patterns are modeled jointly.

### F.8 Sentence-BERT
Sentence-BERT [8] is important for the semantic side of the thesis. The current implementation uses TF-IDF, but the conceptual motivation remains the same: offer descriptions can be embedded and compared semantically. Sentence-BERT is especially useful as a future extension because it could better capture paraphrastic or concept-level similarity than lexical overlap alone.

### F.9 Pcard
Pcard [9] is one of the closest domain analogies to the current work because it uses card transaction history for recommendation. Its importance in the thesis is not limited to the financial domain similarity. It also demonstrates that transaction data can be treated as a recommendation signal in its own right, rather than as a side feature.

### F.10 Transactional Link Prediction
The Scientific Reports paper on link-prediction-based recommendation using transactional data [10] is useful because it expands the methodological horizon beyond standard matrix factorization. It shows that transaction environments can be represented as relational structures and that recommendation can be reframed through graph connectivity.

### F.11 Merchant Recommendation from Credit Card Payments
The Electronics paper on merchant recommendation using credit card payment data [11] is valuable because it highlights domain-specific features. One of the lessons from that work is that recommendation in finance benefits from representations tailored to the domain, not only from generic recommender architectures. This observation supports the thesis choice to build interpretable transaction-derived user profiles instead of jumping directly to opaque neural models.

### F.12 Personalized Time Decay for Financial Products
The personalized time-decay financial recommendation study [12] is probably the single most directly aligned reference for the advanced model in the thesis. It provides direct support for the idea that recency matters in financial recommendation and that user preferences are not stationary. The thesis adapts this intuition in a simpler, more interpretable implementation suitable for the project scope.

### F.13 Synthetic Credit Card Transactions
The work on synthesizing credit card transactions [13] is crucial for the thesis design because it legitimizes the use of synthetic financial data as a research instrument. The current project does not attempt to reproduce all realism dimensions of those studies, but it adopts the broader principle that synthetic transaction environments can support experimentation when privacy blocks access to real data.

### F.14 Why These Sources Matter Together
Taken together, the selected references justify the full project stack. Collaborative filtering explains the baseline interaction perspective. Sequential papers justify recency and future modeling directions. Graph papers justify structural extensions. Hybrid ranking and semantic papers justify the blend of behavioral and textual signals. Domain financial papers justify the relevance of transaction-driven recommendation. Synthetic data papers justify the dataset strategy. This coherence across references is one of the strengths of the thesis and helps demonstrate that the project is not an ad hoc collection of techniques.

## Appendix G. Extended EDA Commentary
### G.1 Why Descriptive Analysis Matters for Recommendation
Exploratory analysis in a recommendation project is not a decorative step that precedes the real machine learning work. In a domain such as banking personalization, EDA directly determines which modeling assumptions are reasonable. If categories are highly concentrated, profile methods become more attractive. If user behavior is temporally volatile, sequence-aware methods become more important. If the offer catalog is semantically broad, textual features become a stronger source of signal. For this reason, the project treats EDA as a substantive analytical chapter rather than a checklist task.

One of the most useful results of the exploratory stage was the confirmation that the synthetic data are structured enough to support ranking experiments. A purely random synthetic generator would not have been useful. The generated dataset displays interpretable category concentration, a realistic spread of spending amounts, and meaningful variation between user segments. Those properties are what make the later recommendation comparison defensible.

Another reason the EDA stage matters is that it frames the interpretation of experimental outcomes. For example, the relatively strong performance of the profile baseline is not surprising once one has seen how category structure organizes user behavior. Likewise, the weak performance of implicit matrix factorization becomes easier to understand when one recalls that the item catalog is small and that the interaction generator is still primarily category-driven.

### G.2 Category Structure and Segment Separation
The top categories by transaction count are transport, restaurants, groceries, entertainment, and education. This distribution is analytically valuable because it combines frequent routine categories with categories that are more event-based or aspiration-oriented. Transport and groceries reflect regular daily life. Entertainment and restaurants represent lifestyle choices and leisure behavior. Education introduces a more purposeful domain that can connect naturally to ecosystem offers or finance products related to self-development.

The presence of education as a strong category is especially important for the thesis because it preserves continuity with the original idea of linking transaction behavior to course recommendations. Even though the thesis topic was later broadened to banking offers in general, the educational dimension still enriches the catalog and makes the hybrid semantic component more meaningful.

Segment separation is also visible in the way categories co-occur. The family segment tends to combine groceries, home, utilities, healthcare, and insurance. The traveler segment combines transport, travel, restaurants, and mobility-related protection. The investor segment stands out through investments and money transfers. The student segment mixes education, entertainment, and transport. These patterns are broad enough to support recommendation, but overlapping enough to avoid trivial one-class-to-one-offer mapping.

This overlap is crucial. If every segment mapped too cleanly to exactly one offer, the recommendation task would reduce to multiclass classification under a thin disguise. Instead, the current design preserves ambiguity and forces the model to rank among several plausible options.

### G.3 Spending Intensity Versus Spending Frequency
A notable analytical lesson from the dataset is that transaction count and transaction value capture different aspects of behavior. Transport and restaurants often generate many events, but individual amounts are moderate. Education, investment, or home-related categories may generate fewer events but with much larger amounts. A profile built only from counts would overemphasize routine but low-value behavior. A profile built only from monetary totals would overemphasize rare but costly events.

The project therefore uses a mixed representation where both spend share and frequency share contribute to the user profile. This design is not only computationally simple but conceptually aligned with banking logic. A bank product manager would likely care both about what a client repeatedly does and about where the client allocates substantial financial resources.

The balance between these two signals also explains why the best time-decay configuration uses spend_weight = 0.60 and freq_weight = 0.40. The benchmark appears to prefer a slight emphasis on monetary intensity while still preserving information about repeated behavioral patterns.

### G.4 Offer Popularity and Benchmark Balance
Offer popularity in the synthetic interaction table is neither uniform nor extremely skewed. Several offers gather similar numbers of positive interactions, with O009, O012, O013, O001, and O002 among the most frequent. This is a useful property for evaluation because it avoids the two extreme failure modes of synthetic benchmarks.

The first failure mode would be near-uniform popularity, where every offer behaves identically and no meaningful ranking signal emerges. The second failure mode would be extreme popularity concentration, where one offer dominates the benchmark and popularity-like heuristics become artificially strong. The current dataset avoids both problems. It has enough variation to create a realistic ranking task, but not so much skew that the comparison becomes trivial.

From the perspective of model behavior, this matters because profile methods are not simply learning to recommend the most popular item. Instead, they must differentiate between multiple offers that are all plausible in different ways.

### G.5 What the EDA Suggests About Future Extensions
The exploratory phase also helps identify which future improvements are most likely to be useful. Because category structure is already informative, future models should not ignore it in favor of purely latent methods. Because recent behavior seems potentially relevant, sequence-aware or attention-based models remain promising. Because the catalog mixes financial and ecosystem products with text descriptions, better semantic embeddings are also justified.

In other words, the EDA stage does not merely describe the dataset; it defines the roadmap. A future version of the project that adds merchant-level simulation, seasonality, and richer product text would make sequential and semantic models more powerful. A future version that adds graph structure between users, categories, merchants, and offers would make graph-based recommenders much more natural. The current thesis therefore treats EDA as both a descriptive and strategic analytical tool.

## Appendix H. Production-Oriented Architecture Roadmap
### H.1 Candidate Generation and Ranking Separation
A production-scale banking recommender would almost certainly use a multi-stage architecture rather than directly scoring every offer with one model. In the current project this complexity is unnecessary because the candidate set contains only fifteen offers. However, from a systems perspective it is useful to describe how the current prototype could evolve.

A common industrial pattern is to separate candidate generation from final ranking. Candidate generation produces a smaller set of plausible offers for each user using fast heuristics, retrieval models, or precomputed embeddings. Final ranking then applies a richer model to reorder those candidates using more features. In the present thesis, profile similarity could play the role of a transparent candidate generator, while a stronger feature-interaction or sequential model could act as the final ranker.

This architecture would be attractive for banking because it preserves both interpretability and scalability. The first stage could guarantee coverage of product families and compliance constraints. The second stage could personalize the final order within a controllable candidate pool.

### H.2 Feature Store and Online-Offline Consistency
A real production system would also require a feature store or an equivalent mechanism for keeping online and offline features consistent. In many industrial machine learning systems, the hardest bugs do not arise from the model itself but from mismatches between training data, batch inference, and real-time inference. For a banking recommender, such mismatches would be especially dangerous because they could result in unstable or inappropriate product recommendations.

The current project already moves in the right direction by structuring feature construction inside reusable pipeline code instead of hidden notebook cells. The next logical step would be to formalize shared feature definitions so that the same spend-share, frequency-share, and time-window logic is reused in both model training and serving.

### H.3 Monitoring and Business KPIs
Offline metrics are necessary for research, but a deployed banking recommender would ultimately be evaluated on business and product KPIs. These could include product view rate, click-through rate, application start rate, product conversion rate, incremental cross-sell revenue, and long-term client value. At the same time, business metrics alone are insufficient if they ignore fairness, risk, or user trust.

A production monitoring stack should therefore combine three layers:
1. system metrics, such as API latency and failure rate;
2. model metrics, such as score drift, calibration shifts, and recommendation coverage;
3. business metrics, such as conversion or downstream engagement.

In a banking setting, one may also need guardrail metrics related to complaints, opt-out rates, or adverse customer reactions. A recommendation that slightly improves click-through but damages trust would not be acceptable.

### H.4 Compliance and Product Governance
Banking recommendation differs from entertainment recommendation because the products carry financial consequences. A music app can tolerate a mildly irrelevant song suggestion. A bank cannot be equally careless when recommending loans, investments, or insurance. For this reason, a future production version of the project would require governance mechanisms beyond pure ranking quality.

Examples include:
- product eligibility rules, so that users do not receive offers for products they cannot access;
- exclusion rules for recently declined or already activated products;
- risk and suitability checks for regulated product families;
- communication caps, so that the same client is not over-targeted across channels.

These governance layers are not implemented in the current prototype, but the thesis discusses them because they are essential for any serious banking deployment.

### H.5 Retraining and Experimentation Policy
Another important production question concerns retraining cadence. A recommendation model trained once and never refreshed will eventually become stale. Banking behavior can change because of seasonality, macroeconomic shifts, changes in product lineup, or life events at the client level. Therefore, a production system needs a retraining policy and an evaluation policy.

One possible roadmap would be:
1. daily or weekly feature refreshes for user profiles;
2. periodic offline evaluation on the latest holdout window;
3. monthly or biweekly retraining of the ranking model;
4. staged deployment of new models through A/B testing.

Such a roadmap balances stability with responsiveness. It also highlights the practical value of the current thesis pipeline: because the project is already script-based and reproducible, it is much closer to such an MLOps transition than a notebook-only prototype would be.

### H.6 Why This Roadmap Matters for the Thesis
Including a production-oriented appendix is useful for two reasons. First, it shows that the project understands the difference between a research benchmark and a deployable system. Second, it strengthens the applied character of the work, which is often important for a master's thesis in data science. The committee does not only want to see metrics; it wants to see that the researcher understands how a model would live inside a real service.

## Appendix I. Extended Threats to Validity
### I.1 Internal Validity
Internal validity concerns whether the observed results actually reflect the compared methods rather than hidden experimental mistakes. The project addresses this by using one evaluation protocol across all models, fixing seeds where appropriate, exporting metrics automatically, and comparing models on the same generated datasets. Nevertheless, risks remain. For example, if a bug in feature computation disproportionately affects one model family, the benchmark could become biased. This is one reason why the thesis emphasizes simple, inspectable baselines and modular pipelines.

### I.2 Construct Validity
Construct validity concerns whether the benchmark reflects the intended real-world problem. The current project aims to model personalization of banking offers from transaction behavior, but several abstractions are unavoidable. MCC-like categories are broader than many real payment taxonomies. The offer catalog is smaller than a real bank's product and campaign space. The interaction generator encodes a simplified notion of relevance. Therefore, the benchmark should be interpreted as a controlled research approximation rather than a complete digital twin of a bank.

### I.3 External Validity
External validity is the most obvious limitation. Results obtained on synthetic data may not transfer quantitatively to a real bank. Real transaction histories include missingness, noise, seasonal shocks, demographic confounds, merchant-specific effects, and strategic marketing exposure. They also reflect organizational policies that shape which clients even get to see which products. The present thesis cannot recreate all those layers. What it can do is provide a transparent framework that is ready to absorb more realistic data when available.

### I.4 Statistical Conclusion Validity
Statistical conclusion validity concerns whether the evidence is strong enough to support the claims. The thesis explicitly addresses this through bootstrap resampling and multi-seed evaluation. These analyses reveal that the gains of the advanced model are modest and not always statistically definitive. This is an important part of the scientific honesty of the project. Rather than hiding ambiguity, the thesis measures it.

### I.5 Practical Validity
A final and often neglected aspect is practical validity: whether the observed metric differences would matter enough to justify system complexity. If a more complicated model improves NDCG@5 only marginally while increasing implementation and maintenance burden substantially, the simpler model may still be preferable in practice. The current results suggest exactly this kind of balanced trade-off. The profile baseline is already strong. Time-decay and hybrid semantics are useful refinements, but their benefits should be judged in relation to engineering cost.

This is why the thesis argues for a staged strategy: begin with interpretable profile-based recommendation, add temporal or semantic refinement where justified, and move to heavier architectures only when data richness and business scale demand it.
