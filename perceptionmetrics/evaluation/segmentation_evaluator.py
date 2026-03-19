class SegmentationEvaluator:
    def __init__(self, model, dataset, n_classes):
        self.model = model
        self.dataset = dataset
        self.metrics = SegmentationMetricsFactory(n_classes)

    def evaluate(self):
        self.metrics.reset()

        for sample in self.dataset:
            image = sample["image"]
            gt = sample["mask"]

            pred = self.model.predict(image)

            self.metrics.update(pred, gt)

        return self.metrics
