relconnector
重新整理的版本

relconnector 核心是把关系表格学习拆分成三个独立、可以并行的部分：

```
sampling（采样）
        -> preparing（数据准备）
                        -> training（训练）
```

对于每个 batch，这三部分必须串行
但是对于不同的 batch，三部分可以并行进行

即可以同时进行 batch1, batch2 的 sampling
甚至可以乱序执行：在 batch1 进行 training 占用 gpu 的时候，不需要等待，可以直接进行 batch2 的数据准备阶段

进一步，这种解耦的设计，对于不同的数据集类型、机器环境，可以通过配置的修改来得到最佳的优化效果：
节省内存的同时、提高训练效率（减少 gpu 空耗）