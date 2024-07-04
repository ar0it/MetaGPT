# 04.07.24
- build the network struc annotations
- prompt engineering for all the prompts!
- the raw data which is read by bioformats still seam to not be the correct one
- think of an interesting metric to compare added annotations
    - Consistency check of the outputs (metric agnostic)
    - Content coverage
    - Task based 
To evaluate the performance of your system that extracts metadata and adds it as structured annotations to the OME Datamodel, without having a ground truth for the structured annotations, you'll need to employ a combination of quantitative and qualitative evaluation methods. Here's an approach you could consider:
- clean up metagpt to be a proper package, restructure to seperate assistant/ predictors/ experiments / evaluators properly.


1. Content Coverage Metric:
   This metric would aim to measure how much of the original metadata is represented in the structured annotations.

   

   ```python
def calculate_content_coverage(original_metadata, structured_annotation):
       total_items = len(original_metadata)
       covered_items = 0
       
       for key, value in original_metadata.items():
           if key in structured_annotation or value in str(structured_annotation):
               covered_items += 1
       
       coverage = covered_items / total_items
       return coverage

   # Example usage
   original_metadata = {
       "Microscope": "Zeiss LSM 880",
       "Objective": "63x oil",
       "Laser Power": "5%",
       "Emission Filter": "BP 500-550"
   }

   structured_annotation = {
       "Equipment": {
           "Microscope": "Zeiss LSM 880",
           "Objective": "63x oil immersion"
       },
       "Settings": {
           "Laser": {
               "Power": "5%"
           },
           "Filter": "Bandpass 500-550 nm"
       }
   }

   coverage = calculate_content_coverage(original_metadata, structured_annotation)
   print(f"Content Coverage: {coverage:.2%}")
   
```

   This metric gives you a quantitative measure of how much of the original metadata is represented in the structured annotations. However, it doesn't account for the accuracy or relevance of the annotations.

2. Semantic Similarity:
   Use embedding models to compute the semantic similarity between the original metadata and the structured annotations. This can help capture whether the essence of the metadata is preserved, even if the exact wording is different.

3. Expert Evaluation:
   Have domain experts review a sample of the structured annotations and rate them on aspects like accuracy, completeness, and relevance. This qualitative feedback can be crucial for understanding the real-world usability of the annotations.

4. Consistency Check:
   Evaluate the consistency of the structured annotations across multiple runs or across similar datasets. This can help identify any instabilities in your LLM-based system.

5. Information Retrieval Metrics:
   Treat the original metadata as queries and the structured annotations as documents. Use metrics like precision, recall, and F1-score to evaluate how well the structured annotations capture the information from the original metadata.

6. Roundtrip Conversion:
   Convert the structured annotations back into a format similar to the original metadata. Then compare this reconstructed metadata with the original using text similarity measures. This can help evaluate if the essential information is preserved through the annotation process.

7. Task-based Evaluation:
   Define specific tasks that rely on the metadata (e.g., image classification, experiment replication) and evaluate how well these tasks can be performed using only the structured annotations compared to using the original metadata.

To implement these evaluation methods, you could create a pipeline that:

1. Extracts a diverse set of metadata from various microscopy image files.
2. Processes this metadata through your LLM-based system to generate structured annotations.
3. Applies the various evaluation metrics and methods described above.
4. Aggregates the results to give an overall performance score.

Here's a high-level pseudocode for such an evaluation pipeline:



```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def extract_metadata(image_file):
    # Implementation to extract metadata from image file
    pass

def generate_structured_annotation(metadata, llm_system):
    # Implementation to generate structured annotation using LLM system
    pass

def calculate_content_coverage(metadata, annotation):
    # Implementation as shown in previous artifact
    pass

def calculate_semantic_similarity(metadata, annotation):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    metadata_embedding = model.encode(str(metadata))
    annotation_embedding = model.encode(str(annotation))
    return cosine_similarity([metadata_embedding], [annotation_embedding])[0][0]

def expert_evaluation(annotation):
    # Implementation to collect and aggregate expert ratings
    pass

def consistency_check(annotations):
    # Implementation to check consistency across multiple annotations
    pass

def information_retrieval_metrics(metadata, annotation):
    # Implementation of precision, recall, and F1-score
    pass

def roundtrip_conversion(metadata, annotation):
    # Implementation to convert annotation back to metadata-like format and compare
    pass

def task_based_evaluation(annotation):
    # Implementation of specific task performance using the annotation
    pass

def evaluate_metadata_extraction(image_files, llm_system):
    results = []
    for file in image_files:
        metadata = extract_metadata(file)
        annotation = generate_structured_annotation(metadata, llm_system)
        
        result = {
            'content_coverage': calculate_content_coverage(metadata, annotation),
            'semantic_similarity': calculate_semantic_similarity(metadata, annotation),
            'expert_score': expert_evaluation(annotation),
            'consistency_score': consistency_check([annotation]),
            'ir_metrics': information_retrieval_metrics(metadata, annotation),
            'roundtrip_similarity': roundtrip_conversion(metadata, annotation),
            'task_performance': task_based_evaluation(annotation)
        }
        results.append(result)
    
    return aggregate_results(results)

def aggregate_results(results):
    # Implementation to aggregate results across all evaluated files
    pass

# Usage
image_files = ['file1.ome.tiff', 'file2.ome.tiff', 'file3.ome.tiff']
llm_system = setup_llm_system()
overall_performance = evaluate_metadata_extraction(image_files, llm_system)
print(f"Overall System Performance: {overall_performance}")

```

This evaluation pipeline combines multiple metrics to give a comprehensive view of your system's performance. The content coverage metric directly addresses your question about how much of the metadata was correctly added to the structured annotation. The semantic similarity and other metrics provide additional context and help evaluate the quality and usefulness of the annotations beyond just content coverage.

To interpret the results:

1. Content coverage will give you a direct measure of how much of the original metadata is represented in some form in the structured annotations.
2. Semantic similarity will help you understand if the meaning of the metadata is preserved, even if the exact wording differs.
3. Expert evaluation provides crucial qualitative feedback on the real-world usability and accuracy of the annotations.
4. Consistency checks help ensure your system is stable and reliable across different inputs.
5. Information retrieval metrics give you a more nuanced view of how well the important information is captured and how easily it can be retrieved from the structured annotations.
6. Roundtrip conversion helps evaluate if the essential information is preserved through the annotation process.
7. Task-based evaluation provides insight into the practical utility of the structured annotations.

By combining these metrics, you can get a comprehensive understanding of your system's performance in extracting and structuring metadata, even without a direct ground truth for the structured annotations.

Would you like me to elaborate on any specific part of this evaluation approach or discuss how to implement any of these metrics in more detail?


# 03.07.24
- build the analysis pipeline for the struc annotation track and run it 
- the raw data which is read by bioformats still seam to not be the correct one
- network of for struc annotations
- tree swarm for struc annotations

# 02.07.4
- Structured Annotations track:
    - try to improve the one shot results via prompt enigneering.
    - try the predictor tree, with structured annotations as root
    - try a multi step predictor which first isolatest non OME data and then creates the structured annotation

# 28.06.24
- Structured annotation track:
    - 

# 27.06.24
- General Things:
    - Dont spend more than 500€ for all things combined
- Structured annotation track:
    - no validation required here (maybe )
    - start with single marvin agents with empty state
    - continue with network approach (first sort data in data which is not present in autogenerated OME XML)
    - ?? do the fancy graph approach?

- Qualitative Results track:
    - show the results for the singular agent on marvin agent + ome schema for very small test metadata
    - ... on network for very small test metadata
    - show the results for the very small test metadata for the graph approach
    

- Quantitative Results track:

# 20.06.24
- the state incorporate the lower lvvl states which have alrdy been predicted, but the llm doesnt know that and tries to predict those again maybe fixed by deleting added raw_meta, also need to restructure prompt probably
- use the same assistant/ thread for all predictions?
- MaybeModels create problems since the subnodes are not added currently
- Model requiring another object for instantiation might be only relevant for image/pixels 

# 19.06.24
- what to do about nodes that can occur multiple times (in the tree model)
- Remove metadata already implemented from raw_meta ?
- unrecognized fields

# 16.06.24
- Limit the tokens used for a request. (I want control over the cost)
- Implement code async?

# 14.06.24

**ToDos**
- Maybe develop a heruistic on how to hand the metadata to the agent?
    - ideas:
    - sort by key length (proportional to nestedness)
        - alsways provide ~ 10 keys at once

    - sort by count of "|" as this is a common delimiter for the path in keys (proportional for nestedness)
    - have an llm one shot the order?
    - try to build a graph from the "|" structure

- maybe give the proprietary file format as input aswell
- maybe compress error messages handed to gpt4o via a gpt3.5 instance
- funny idea for are llms intelligent --> give a base task i.e. write a text. then recursively simply ask "improve this" do the same with humans. How does the curve look like for a plot where x is the iterations and y is the quality of the response.

- Idea for swarm: 1 agent doesn the schema focused retrieval, 1 agent who does the raw data retrieval, 1 Agent who asks questions and 1 agent who does the tool usage.
- Question answering seems to limit the scope, and might be more reliable than free responses
- Structure the OME schema into a questionair which the ai models chats with.
# 13.06.24

**Known Bugs/ToDos remaining:**

---

# 12.06.24

**Known Bugs/Todos remaining:**
- ~~Fix bugs with the reading of the raw metadata~~
- ~~provide a starting state to marvin~~
- iteratively update the state to validate more regularly (need more concrete feedback)
- generate some nicer figures, approach the figure business more systematically
- create some red line for the thesis, which questions need to be answered

**Thoughts**
- There are some metadata foramts whihc encrypt their data to sth like unit: 8
where 8 is by definition mm. This will be hard for an LLM to know
---