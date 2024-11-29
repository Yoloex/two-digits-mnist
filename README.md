# Two digits MNIST classification

This repo handles two-digits MNIST classification as a multilabel one.

## Solution

### Data Analysis

Take a look at `data_explore.ipynb`.

### Train, save and test model

Refer to `main.ipynb` and run `predict_test.py`.

### Export model

`export.py` will export torch model to `onnxruntime`

### Serve model

Take a look at `serve.py`

## Answer to Question 3

To deploy the model as a service to process the influx of 100 images per second, I could use a cloud platform like AWS or Google Cloud.

I will leverage scalable compute resources such as Kubernetes for orchestration.

I can set up an API endpoint that utilizes a fast image processing pipeline, possibly integrating image classification or analysis. To handle the high volume, can consider batching requests or using a streaming framework like Apache Kafka, combined with autoscaling to ensure optimal resource allocation and minimal latency.
