# v2 两组对比总结

新增第20页，四张架构图顺延至第21–24页。

本页使用历史报告与先前选定 v2 实测记录，未使用内存估算。
{
  "aggregation": "equal-weight arithmetic mean of per-dataset relative ratios",
  "rss_dataset_count": 10,
  "vanilla_time_datasets": [
    "rel-event",
    "rel-f1",
    "rel-salt",
    "rel-stack",
    "rel-trial"
  ],
  "batch_time_dataset_count": 10,
  "vanilla_rss_record_reduction_pct": 27.49722668679304,
  "vanilla_time_reduction_pct": 64.4197293580275,
  "batch_rss_record_over_v2_main_rss_ratio": 9.181886986380121,
  "v2_over_batch_time_ratio": 1.75316131644893,
  "v2_over_batch_time_increase_pct": 75.31613164489299,
  "warning": "RSS records are not comparable total-process-tree memory: v2 only RUSAGE_SELF. Time is end-to-end seconds/batch, includes preparation and mixed cache states; Vanilla time excludes five timeout datasets."
}
公式：Vanilla RSS = mean(1-v2_RSS_i/vanilla_RSS_i)；Vanilla 耗时 = mean(1-v2_sec_per_batch_i/vanilla_sec_per_batch_i)，仅5个完成项；Batch RSS 倍数 = mean(batch_RSS_i/v2_main_RSS_i)；Batch 耗时增加 = mean(v2_sec_per_batch_i/batch_sec_per_batch_i-1)。
平均比值与均值之比不同。本页 Batch 耗时1.753×是数据集等权平均；旧页按实际batch数量折算的合计比值2.300×是另一种加权方式，不应混用。RSS不支持整体内存优化的因果结论。
来源：doc/gpu-feature-optimization-experiment.md；doc/slides/bottleneck-review-20260927/rss-data.csv；doc/slides/bottleneck-review-20260927/performance-data.json；relconnector_v2/relconnector_v2/benchmark.py:98。
