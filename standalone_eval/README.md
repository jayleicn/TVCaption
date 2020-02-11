TVC Evaluation
================================================================

### Task Definition
Given a video (with subtitle), a pair of timestamps that specifies a moment inside the video, 
the TVC task requires a system to comprehend both the video and the subtitle to generate a 
description of the specified moment. 

### How to construct a prediction file?

An example of such file is [sample_val_predictions.jsonl](sample_val_predictions.jsonl), 
each line of this file is a json object, formatted as:
```
{
    "vid_name": "friends_s01e03_seg02_clip_19", 
    "clip_id": 86603, 
    "ts": [16.48, 33.87], 
    "descs": [{"desc": "phoebe and monica are having a conversation with each other ."}]
}
``` 

### Run Evaluation
At project root, run
```
bash standalone_eval/eval_sample.sh 
```
This command will use [evaluate.py](evaluate.py) to evaluate 
[sample_val_predictions.jsonl](sample_val_predictions.jsonl), 
the output will be written into `sample_val_predictions_metrics.json`. 
Its content should be similar if not the same as `sample_val_predictions_metrics_raw.json` file.


### Codalab Submission
To test your model's performance on `test-public` set, 
please submit both `val` and `test-public` predictions to our 
[Codalab evaluation server](https://competitions.codalab.org/competitions/23109). 
The submission file should be a single `.zip ` file (no enclosing folder) 
that contains the two prediction files 
`tvc_test_public_submission.jsonl` and `tvc_val_submission.jsonl`, each of the `*submission.jsonl` file 
should be formatted as instructed above. 
