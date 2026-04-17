# Sources Review (Checkpoint 2)

Date: 2026-04-04  
Topic: personalization of customer offers based on banking transaction activity.

## 1) Краткая суммаризация
- Для персонализации банковских предложений наиболее релевантны четыре класса подходов:
  1. implicit-feedback collaborative filtering;
  2. feature-interaction / CTR-style models;
  3. sequential recommendation;
  4. transaction-graph and domain-specific merchant recommendation.
- В прикладной постановке ВКР ключевыми оказываются не только модели, но и протокол оценки:
  `Precision@K`, `Recall@K`, `MAP@K`, `NDCG@K`; в доменных работах также встречаются `HR@K`, `AUC`, pairwise ranking accuracy.
- По литературе видно, что для транзакционных данных особенно важны три источника сигнала:
  1. долгосрочный профиль расходов;
  2. недавняя динамика и порядок событий;
  3. доменные признаки объекта рекомендации: категория, текст, регион, merchant metadata.
- Для данной ВКР разумная структура выглядит так:
  - baseline: profile similarity, item-based CF, implicit MF, NCF;
  - stronger branch: SASRec/BERT4Rec/TiSASRec или LightGCN;
  - semantic layer: sentence embeddings для текстов офферов;
  - practical layer: воспроизводимый pipeline и сервисный прототип.

## 2) Что берем в диплом уже сейчас
- Метрики: `Precision@K`, `Recall@K`, `MAP@K`, `NDCG@K` как основные для top-K выдачи.
- Базовые модели: implicit MF и NCF как академически узнаваемые baseline-опоры.
- SOTA-ориентиры: SASRec, BERT4Rec, TiSASRec, LightGCN.
- Аналоги в транзакционном домене: Pcard, link prediction on transactional data, merchant recommender on credit card payments.
- Инженерные идеи: Wide & Deep, DeepFM, xDeepFM как ориентиры для feature-rich ranking stage.
- Репродуцируемые артефакты: сравнительные таблицы моделей, схема pipeline, архитектурная схема сервиса, графики метрик и sensitivity/robustness analysis.

## 3) Сравнительная таблица источников
| № | Источник | Год / площадка | Тип | Метрики / протокол | Что переиспользуем |
|---|---|---|---|---|---|
| 1 | [Neural Collaborative Filtering](https://doi.org/10.1145/3038912.3052569) | 2017, WWW | implicit CF | ranking comparison on implicit data | Нелинейный baseline |
| 2 | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) | 2016, Google / DLRS | industrial ranking | online recommendation setup | Архитектурный ориентир для сервиса |
| 3 | [DeepFM](https://www.ijcai.org/proceedings/2017/0239.pdf) | 2017, IJCAI | feature interaction | CTR / ranking style evaluation | Кандидат для tabular ranking stage |
| 4 | [xDeepFM](https://arxiv.org/abs/1803.05170) | 2018, KDD | feature interaction | benchmark comparison | Идея явных и неявных feature interactions |
| 5 | [SASRec](https://arxiv.org/abs/1808.09781) | 2018, ICDM | sequential recsys | HR/NDCG on public datasets | SOTA-ветка по порядку транзакций |
| 6 | [BERT4Rec](https://arxiv.org/abs/1904.06690) | 2019, CIKM | bidirectional sequential recsys | HR/NDCG on four benchmarks | Усиление sequential branch |
| 7 | [TiSASRec](https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm20b.pdf) | 2020, WSDM | time-aware sequential recsys | ranking metrics with time intervals | Теоретическое обоснование time-aware логики |
| 8 | [LightGCN](https://arxiv.org/abs/2002.02126) | 2020, SIGIR | graph recsys | Recall/NDCG vs graph baselines | Графовый кандидат для user-offer graph |
| 9 | [Sentence-BERT](https://aclanthology.org/D19-1410/) | 2019, EMNLP-IJCNLP | semantic text embeddings | STS / transfer evaluation | Семантический слой для текстов офферов |
| 10 | [Pcard: Personalized Restaurants Recommendation from Card Payment Transaction Records](https://users.cs.utah.edu/~lifeifei/papers/pcard-www19.pdf) | 2019, WWW | card transaction recommendation | pairwise and ranking evaluation | Прямая карточно-транзакционная аналогия |
| 11 | [A Link Prediction-Based Recommendation System Using Transactional Data](https://www.nature.com/articles/s41598-023-34055-5) | 2023, Scientific Reports | transaction graph | AUC, MAP@K | Аналог для graph-based evaluation |
| 12 | [Merchant Recommender System Using Credit Card Payment Data](https://www.mdpi.com/2079-9292/12/4/811) | 2023, Electronics | merchant recommendation | HR@5/10/20, NDCG@5/10/20 | Domain features for banking-like setup |
| 13 | [Recommendation System with Minimized Transaction Data](https://www.sciencedirect.com/science/article/pii/S2666764922000017) | 2022, Data Science and Management | privacy-aware transaction recsys | quality under reduced schema | Аргумент про минимальный набор transaction features |
| 14 | [Synthesizing Credit Card Transactions](https://arxiv.org/abs/1910.03033) | 2019, arXiv | synthetic financial data | realism-oriented validation | Обоснование synthetic data strategy |
| 15 | [Deep Learning Based Recommender System: A Survey and New Perspectives](https://doi.org/10.1145/3285029) | 2019, ACM CSUR | survey | систематизация классов моделей | Общая рамка и позиционирование ВКР |

## 4) Тематические выводы по источникам

### 4.1 Baseline-модели и минимально достаточный стек
1. **NCF** стоит использовать как обязательный нелинейный baseline.
Причина: работа узнаваемая, академически сильная и хорошо ложится на implicit-feedback постановку.

2. **Wide & Deep, DeepFM, xDeepFM** полезны не как единственный финальный выбор, а как ориентир для инженерной части.
Причина: они хорошо объясняют, как соединять sparse tabular features, memorization/generalization и взаимодействия признаков в production-like ranking.

3. **Implicit MF** по литературе остается сильным и честным baseline.
Причина: во многих транзакционных задачах простая матричная факторизация оказывается устойчивее сложных нейросетевых моделей при ограниченном объеме или шумности данных.

### 4.2 Sequential и time-aware SOTA
4. **SASRec** показывает, что сам порядок событий пользователя является полезным сигналом.
Что берем: идею последовательной модели как следующего логичного апгрейда после классических baseline.

5. **BERT4Rec** усиливает sequential recommendation за счет bidirectional encoding.
Что берем: аргумент, что masked sequence modeling может быть естественным следующим этапом для транзакционных последовательностей.

6. **TiSASRec** особенно важен именно для текущей темы.
Что берем: прямое теоретическое подтверждение, что интервалы времени между событиями улучшают моделирование поведения; это хорошо поддерживает используемую в работе time-decay ветку.

### 4.3 Graph и transaction-domain аналоги
7. **LightGCN** полезен как чистый graph baseline без лишней архитектурной сложности.
Что берем: если на защите спросят про альтернативу последовательным моделям, LightGCN выглядит самым понятным графовым кандидатом.

8. **Pcard** является самой близкой прикладной аналогией из открытых источников.
Что берем: связку card transactions + personalized ranking; это сильный аргумент, что сама постановка ВКР не искусственна.

9. **Scientific Reports 2023** и **Electronics 2023** подтверждают, что транзакционные данные пригодны для рекомендации даже вне классического e-commerce catalog recommendation.
Что берем: графовую постановку, доменные признаки мерчанта, а также список метрик, которые выглядят убедительно в транзакционном домене.

10. **Recommendation System with Minimized Transaction Data** полезна как контраргумент на вопрос о неполных банковских данных.
Что берем: мысль о том, что даже ограниченная transaction schema может быть полезной для персонализации, если аккуратно выбрать признаки.

### 4.4 Данные, воспроизводимость и ограничения
11. **Synthesizing Credit Card Transactions** поддерживает саму логику synthetic data в исследовании.
Что берем: аргументацию, что синтетические финансовые данные могут быть оправданным компромиссом при закрытости реальных логов.

12. **Survey paper** нужен не ради конкретной модели, а ради рамки.
Что берем: понятный язык позиционирования ВКР между collaborative, content-aware, sequential и graph-based recommendation.

## 5) Что важно показать научному руководителю
- Обзор литературы не должен выглядеть как набор пересказов.
Нужно явно показать, почему выбраны именно такие baseline-модели, почему time-aware ветка разумна и почему переход к sequential/graph SOTA выглядит методологически оправданным.

- Для диплома важно честно развести две линии:
  1. что уже реализовано и проверено;
  2. что является научно обоснованным следующим шагом.

- Сильная формула позиционирования для ВКР:
  воспроизводимый banking-like pipeline + сравнение академически понятных baseline-моделей + time-aware extension + sanity check на реальном transaction log.

- Для презентации наиболее полезные артефакты по литературе:
  - одна сравнительная таблица моделей;
  - одна схема pipeline;
  - один слайд про gap: закрытость банковских данных и редкость открытых transaction-domain benchmarks.

## 6) Рекомендованный shortlist для текста ВКР
- Обязательно оставить в основном тексте: 1, 5, 6, 7, 8, 9, 10, 11, 12, 14.
- В инженерной части удобно использовать: 2, 3, 4.
- Для общего позиционирования и введения: 15.

## 7) Итоговый вывод
По состоянию на апрель 2026 года обзор источников подтверждает, что выбранная тема корректно лежит на пересечении recommender systems, transaction analytics и practical ML engineering. В открытой литературе нет большого числа полностью воспроизводимых банковских кейсов, зато есть достаточная база из фундаментальных recsys-работ, time-aware/sequential SOTA и нескольких близких transaction-domain статей. Это позволяет обосновать текущую архитектуру ВКР и одновременно честно показать, где заканчивается уже реализованная часть и начинается следующий исследовательский шаг.
