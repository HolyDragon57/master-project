### VideoMAE => ActionCLIP => VideoLLaVA

肯定都是需要根据标注数据专门训练

VideoLLaVA还不太清楚如何涉及动作分类的训练

大部分需要降采样，可能要以短边优先保证aspect ratio切割。