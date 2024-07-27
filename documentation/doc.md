---
description: |
    API documentation for modules: metagpt, metagpt.distorters, metagpt.distorters.distorter_template, metagpt.evaluators, metagpt.evaluators.evaluator_template, metagpt.experiments, metagpt.experiments.experiment_template, metagpt.predictors, metagpt.predictors.predictor_distorter, metagpt.predictors.predictor_network, metagpt.predictors.predictor_network_annotator, metagpt.predictors.predictor_seperator, metagpt.predictors.predictor_simple, metagpt.predictors.predictor_simple_annotator, metagpt.predictors.predictor_state, metagpt.predictors.predictor_state_tree, metagpt.predictors.predictor_template, metagpt.utils, metagpt.utils.BioformatsReader, metagpt.utils.DataClasses, metagpt.utils.utils.

lang: en

classoption: oneside
geometry: margin=1in
papersize: a4

linkcolor: blue
links-as-notes: true
...


    
# Module `metagpt` {#id}




    
## Sub-modules

* [metagpt.distorters](#metagpt.distorters)
* [metagpt.evaluators](#metagpt.evaluators)
* [metagpt.experiments](#metagpt.experiments)
* [metagpt.predictors](#metagpt.predictors)
* [metagpt.utils](#metagpt.utils)






    
# Namespace `metagpt.distorters` {#id}




    
## Sub-modules

* [metagpt.distorters.distorter_template](#metagpt.distorters.distorter_template)






    
# Module `metagpt.distorters.distorter_template` {#id}

This module contains the DistorterTemplate class, which is responsible for distorting
well-formed OME XML into a modified key-value representation. The distortion process
can include converting XML to key-value pairs, shuffling the order of entries, and
renaming keys to similar words.





    
## Classes


    
### Class `DistorterTemplate` {#id}




>     class DistorterTemplate


A class for distorting OME XML into modified key-value representations.

The distorter takes well-formed OME XML as input and returns a "distorted"
key-value version of it. Distortion can include:
- OME XML to key-value conversion
- Shuffling of the order of entries
- Renaming keys to similar words







    
#### Methods


    
##### Method `distort` {#id}




>     def distort(
>         self,
>         ome_xml: str,
>         out_path: str,
>         should_pred: str = 'maybe'
>     ) ‑> Optional[Dict[str, Any]]


Distort the OME XML.


Args
-----=
**```ome_xml```** :&ensp;<code>str</code>
:   The input OME XML string.


**```out_path```** :&ensp;<code>str</code>
:   The path where the distorted data will be saved.


**```should_pred```** :&ensp;<code>str</code>
:   Whether to predict new data or use existing data. Options are "yes", "no", or "maybe".



Returns
-----=
<code>Optional\[Dict\[str, Any]]</code>
:   The distorted metadata, or None if no data is available.



    
##### Method `extract_unique_keys` {#id}




>     def extract_unique_keys(
>         self,
>         metadata: Dict[str, Any]
>     ) ‑> List[str]


Extract all unique key names from a dictionary, including nested structures,
without full paths or indices.


Args
-----=
**```metadata```** :&ensp;<code>Dict\[str, Any]</code>
:   The dictionary containing metadata.



Returns
-----=
<code>List\[str]</code>
:   A list of unique key names.



    
##### Method `gen_mapping` {#id}




>     def gen_mapping(
>         self,
>         dict_meta: Dict[str, Any]
>     ) ‑> Dict[str, str]


Rename the keys in the OME XML to similar words using a GPT model.


Args
-----=
**```dict_meta```** :&ensp;<code>Dict\[str, Any]</code>
:   The input dictionary.



Returns
-----=
<code>Dict\[str, str]</code>
:   A dictionary mapping original keys to new keys.



    
##### Method `isolate_keys` {#id}




>     def isolate_keys(
>         self,
>         dict_meta: Dict[str, Any]
>     ) ‑> Dict[str, None]


Isolate the keys in the OME XML.


Args
-----=
**```dict_meta```** :&ensp;<code>Dict\[str, Any]</code>
:   The input dictionary.



Returns
-----=
<code>Dict\[str, None]</code>
:   A dictionary with the same keys as the input, but all values set to None.



    
##### Method `load_fake_data` {#id}




>     def load_fake_data(
>         self,
>         path: str
>     ) ‑> Optional[Dict[str, Any]]


Load the fake data from a file.


Args
-----=
**```path```** :&ensp;<code>str</code>
:   The file path from which to load the data.



Returns
-----=
<code>Optional\[Dict\[str, Any]]</code>
:   The loaded data, or None if the file doesn't exist.



    
##### Method `modify_metadata_structure` {#id}




>     def modify_metadata_structure(
>         self,
>         metadata: Dict[str, Any],
>         operations: Optional[List[<built-in function callable>]] = None,
>         probability: float = 0.3
>     ) ‑> Dict[str, Any]


Modify the structure of a metadata dictionary systematically and randomly.


Args
-----=
**```metadata```** :&ensp;<code>Dict\[str, Any]</code>
:   The original metadata dictionary.


**```operations```** :&ensp;<code>Optional\[List\[callable]]</code>
:   List of operations to perform. If None, all operations are used.


**```probability```** :&ensp;<code>float</code>
:   Probability of applying an operation to each element (0.0 to 1.0).



Returns
-----=
<code>Dict\[str, Any]</code>
:   A new dictionary with modified structure.



    
##### Method `pred` {#id}




>     def pred(
>         self,
>         ome_xml: str,
>         out_path: str
>     ) ‑> Dict[str, Any]


Predict the distorted data.


Args
-----=
**```ome_xml```** :&ensp;<code>str</code>
:   The input OME XML string.


**```out_path```** :&ensp;<code>str</code>
:   The path where the distorted data will be saved.



Returns
-----=
<code>Dict\[str, Any]</code>
:   The distorted metadata.



    
##### Method `rename_metadata_keys` {#id}




>     def rename_metadata_keys(
>         self,
>         metadata: Dict[str, Any],
>         key_mapping: Dict[str, str]
>     ) ‑> Dict[str, Any]


Rename keys in a metadata dictionary based on a provided mapping.


Args
-----=
**```metadata```** :&ensp;<code>Dict\[str, Any]</code>
:   The original metadata dictionary.


**```key_mapping```** :&ensp;<code>Dict\[str, str]</code>
:   A dictionary mapping original key names to new key names.



Returns
-----=
<code>Dict\[str, Any]</code>
:   A new dictionary with renamed keys.



    
##### Method `save_fake_data` {#id}




>     def save_fake_data(
>         self,
>         fake_data: Dict[str, Any],
>         path: str
>     ) ‑> None


Save the fake data to a file.


Args
-----=
**```fake_data```** :&ensp;<code>Dict\[str, Any]</code>
:   The data to be saved.


**```path```** :&ensp;<code>str</code>
:   The file path where the data will be saved.



    
##### Method `shuffle_order` {#id}




>     def shuffle_order(
>         self,
>         dict_meta: Dict[str, Any]
>     ) ‑> Dict[str, Any]


Shuffle the order of the keys in the OME XML.


Args
-----=
**```dict_meta```** :&ensp;<code>Dict\[str, Any]</code>
:   The input dictionary.



Returns
-----=
<code>Dict\[str, Any]</code>
:   A new dictionary with shuffled keys.



    
##### Method `xml_to_key_value` {#id}




>     def xml_to_key_value(
>         self,
>         ome_xml: str
>     ) ‑> Dict[str, Any]


Convert the OME XML to key-value pairs.


Args
-----=
**```ome_xml```** :&ensp;<code>str</code>
:   The input OME XML string.



Returns
-----=
<code>Dict\[str, Any]</code>
:   A dictionary representation of the XML.



Raises
-----=
<code>Exception</code>
:   If parsing fails.





    
# Namespace `metagpt.evaluators` {#id}




    
## Sub-modules

* [metagpt.evaluators.evaluator_template](#metagpt.evaluators.evaluator_template)






    
# Module `metagpt.evaluators.evaluator_template` {#id}

This module contains the EvaluatorTemplate class, which is responsible for evaluating
the performance of OME XML generation models by calculating the edit distance between
the ground truth and the prediction.

The class provides various methods for data analysis and visualization, including
edit distance calculations, path analysis, and performance comparisons across
different methods and image formats.





    
## Classes


    
### Class `EvaluatorTemplate` {#id}




>     class EvaluatorTemplate(
>         schema: Optional[str] = None,
>         dataset: Optional[metagpt.utils.DataClasses.Dataset] = None,
>         out_path: Optional[str] = None
>     )


This class evaluates the performance of an OME XML generation model by calculating
the edit distance between the ground truth and the prediction.

Reference: <https://github.com/timtadh/zhang-shasha>

Initialize the EvaluatorTemplate.


Args
-----=
**```schema```** :&ensp;<code>Optional\[str]</code>
:   The schema to use for evaluation.


**```dataset```** :&ensp;<code>Optional\[Dataset]</code>
:   The dataset to evaluate.


**```out_path```** :&ensp;<code>Optional\[str]</code>
:   The output path for saving results.









    
#### Methods


    
##### Method `attempts_paths_plt` {#id}




>     def attempts_paths_plt(
>         self,
>         df_sample: pandas.core.frame.DataFrame
>     )


Plot number of attempts against number of paths.

    
##### Method `format_counts_plt` {#id}




>     def format_counts_plt(
>         self,
>         df_sample: pandas.core.frame.DataFrame
>     )


Plot counts of samples by image format.

    
##### Method `format_method_plt` {#id}




>     def format_method_plt(
>         self,
>         df_sample: pandas.core.frame.DataFrame
>     )


Plot edit distance by method and image format.

    
##### Method `generate_results_report` {#id}




>     def generate_results_report(
>         self,
>         figure_paths: List[str],
>         context: str
>     ) ‑> Optional[str]


Generate a results report based on the provided figures and context.


Args
-----=
**```figure_paths```** :&ensp;<code>List\[str]</code>
:   Paths to the figure images.


**```context```** :&ensp;<code>str</code>
:   Context information for the report.



Returns
-----=
<code>Optional\[str]</code>
:   The generated report, or None if an error occurred.



    
##### Method `get_graph` {#id}




>     def get_graph(
>         self,
>         xml_root: ome_types._autogenerated.ome_2016_06.ome.OME,
>         root: Optional[zss.simple_tree.Node] = None
>     ) ‑> zss.simple_tree.Node


Get the graph representation of an OME XML tree as a zss Node.


Args
-----=
**```xml_root```** :&ensp;<code>OME</code>
:   The root of the XML tree.


**```root```** :&ensp;<code>Optional\[Node]</code>
:   The root node of the graph (used for recursion).



Returns
-----=
<code>Node</code>
:   The root node of the graph representation.



    
##### Method `json_to_pygram` {#id}




>     def json_to_pygram(
>         self,
>         json_data: Dict[str, Any]
>     ) ‑> Any


Convert a JSON structure to a pygram tree.


Args
-----=
**```json_data```** :&ensp;<code>Dict\[str, Any]</code>
:   The JSON data to convert.



Returns
-----=
<code>Any</code>
:   The root node of the pygram tree.



    
##### Method `method_attempts_plt` {#id}




>     def method_attempts_plt(
>         self,
>         df_sample: pandas.core.frame.DataFrame
>     )


Plot number of attempts by method.

    
##### Method `method_cost_plt` {#id}




>     def method_cost_plt(
>         self,
>         df_sample: pandas.core.frame.DataFrame
>     )


Plot method cost.

    
##### Method `method_edit_distance_no_annot_plt` {#id}




>     def method_edit_distance_no_annot_plt(
>         self,
>         df_sample: pandas.core.frame.DataFrame
>     )


Plot method edit distance without annotations.

    
##### Method `method_edit_distance_only_annot_plt` {#id}




>     def method_edit_distance_only_annot_plt(
>         self,
>         df_sample: pandas.core.frame.DataFrame
>     )


Plot method edit distance for annotations only.

    
##### Method `method_edit_distance_plt` {#id}




>     def method_edit_distance_plt(
>         self,
>         df_sample: pandas.core.frame.DataFrame
>     )


Plot method edit distance.

    
##### Method `method_time_plt` {#id}




>     def method_time_plt(
>         self,
>         df_sample: pandas.core.frame.DataFrame
>     )


Plot method prediction time.

    
##### Method `n_paths_cost_plt` {#id}




>     def n_paths_cost_plt(
>         self,
>         df_sample: pandas.core.frame.DataFrame
>     )


Plot number of paths against cost.

    
##### Method `n_paths_method_plt` {#id}




>     def n_paths_method_plt(
>         self,
>         df_sample: pandas.core.frame.DataFrame
>     )


Plot number of paths per method.

    
##### Method `n_paths_time_plt` {#id}




>     def n_paths_time_plt(
>         self,
>         df_sample: pandas.core.frame.DataFrame
>     )


Plot number of paths against prediction time.

    
##### Method `path_df` {#id}




>     def path_df(
>         self
>     ) ‑> pandas.core.frame.DataFrame


Create a DataFrame with paths as Index and samples as Columns.


Returns
-----=
<code>pd.DataFrame</code>
:   DataFrame with path information.



    
##### Method `path_difference` {#id}




>     def path_difference(
>         self,
>         xml_a: ome_types._autogenerated.ome_2016_06.ome.OME,
>         xml_b: ome_types._autogenerated.ome_2016_06.ome.OME
>     ) ‑> int


Calculate the length of the difference between the path sets in two XML trees.


Args
-----=
**```xml_a```** :&ensp;<code>OME</code>
:   The first XML tree.


**```xml_b```** :&ensp;<code>OME</code>
:   The second XML tree.



Returns
-----=
<code>int</code>
:   The length of the difference between the path sets.



    
##### Method `paths_annotation_stacked_plt` {#id}




>     def paths_annotation_stacked_plt(
>         self,
>         df_sample: pandas.core.frame.DataFrame
>     )


Plot stacked bar chart of paths and annotations.

    
##### Method `plot_price_per_token` {#id}




>     def plot_price_per_token(
>         self
>     )


Plot price per token for different models.

    
##### Method `pygram_edit_distance` {#id}




>     def pygram_edit_distance(
>         self,
>         xml_a: ome_types._autogenerated.ome_2016_06.ome.OME,
>         xml_b: ome_types._autogenerated.ome_2016_06.ome.OME
>     ) ‑> float


Calculate the edit distance between two XML trees using pygram.


Args
-----=
**```xml_a```** :&ensp;<code>OME</code>
:   The first XML tree.


**```xml_b```** :&ensp;<code>OME</code>
:   The second XML tree.



Returns
-----=
<code>float</code>
:   The edit distance between the two trees.



    
##### Method `report` {#id}




>     def report(
>         self
>     )


Generate and write an evaluation report to a file.

    
##### Method `sample_df` {#id}




>     def sample_df(
>         self,
>         df_paths: pandas.core.frame.DataFrame
>     ) ‑> pandas.core.frame.DataFrame


Create a DataFrame with samples as Index and properties as Columns.


Args
-----=
**```df_paths```** :&ensp;<code>pd.DataFrame</code>
:   DataFrame containing path information.



Returns
-----=
<code>pd.DataFrame</code>
:   DataFrame with sample properties.



    
##### Method `zss_edit_distance` {#id}




>     def zss_edit_distance(
>         self,
>         xml_a: ome_types._autogenerated.ome_2016_06.ome.OME,
>         xml_b: ome_types._autogenerated.ome_2016_06.ome.OME
>     ) ‑> int


Calculate the Zhang-Shasha edit distance between two XML trees.


Args
-----=
**```xml_a```** :&ensp;<code>OME</code>
:   The first XML tree.


**```xml_b```** :&ensp;<code>OME</code>
:   The second XML tree.



Returns
-----=
<code>int</code>
:   The edit distance between the two trees.





    
# Namespace `metagpt.experiments` {#id}




    
## Sub-modules

* [metagpt.experiments.experiment_template](#metagpt.experiments.experiment_template)






    
# Module `metagpt.experiments.experiment_template` {#id}

This module contains the ExperimentTemplate class, which defines an experiment object
that can be used to run experiments. The experiment defines the dataset, predictors,
evaluators, and other parameters necessary for running experiments on OME XML metadata.





    
## Classes


    
### Class `ExperimentTemplate` {#id}




>     class ExperimentTemplate


The ExperimentTemplate class defines an experiment object that can be used to run experiments.
It encapsulates the dataset, predictors, evaluators, and other experiment parameters.

Initialize the ExperimentTemplate with default values.







    
#### Methods


    
##### Method `run` {#id}




>     def run(
>         self
>     ) ‑> None


Run the experiment.
This method processes each image in the data_paths, generates metadata,
and runs predictors and evaluators.



    
# Module `metagpt.predictors` {#id}




    
## Sub-modules

* [metagpt.predictors.predictor_distorter](#metagpt.predictors.predictor_distorter)
* [metagpt.predictors.predictor_network](#metagpt.predictors.predictor_network)
* [metagpt.predictors.predictor_network_annotator](#metagpt.predictors.predictor_network_annotator)
* [metagpt.predictors.predictor_seperator](#metagpt.predictors.predictor_seperator)
* [metagpt.predictors.predictor_simple](#metagpt.predictors.predictor_simple)
* [metagpt.predictors.predictor_simple_annotator](#metagpt.predictors.predictor_simple_annotator)
* [metagpt.predictors.predictor_state](#metagpt.predictors.predictor_state)
* [metagpt.predictors.predictor_state_tree](#metagpt.predictors.predictor_state_tree)
* [metagpt.predictors.predictor_template](#metagpt.predictors.predictor_template)






    
# Module `metagpt.predictors.predictor_distorter` {#id}

This module contains the PredictorDistorter class, which is responsible for
inventing new metadata syntax for microscopy images based on existing metadata.





    
## Classes


    
### Class `PredictorDistorter` {#id}




>     class PredictorDistorter(
>         raw_meta: str
>     )


A predictor class for inventing new metadata syntax for microscopy images.

This class takes existing metadata and translates it into a new syntax,
maintaining the original structure and values but changing the keys.

Initialize the PredictorDistorter.


Args
-----=
**```raw_meta```** :&ensp;<code>str</code>
:   The raw metadata to be translated.




    
#### Ancestors (in MRO)

* [metagpt.predictors.predictor_template.PredictorTemplate](#metagpt.predictors.predictor_template.PredictorTemplate)



    
#### Class variables


    
##### Variable `out_new_meta` {#id}




Helper class to define the output structure of the assistant.




    
#### Methods


    
##### Method `init_assistant` {#id}




>     def init_assistant(
>         self
>     ) ‑> None


Initialize the OpenAI assistant.

    
##### Method `init_run` {#id}




>     def init_run(
>         self
>     ) ‑> None


Initialize and monitor the run of the assistant.

    
##### Method `predict` {#id}




>     def predict(
>         self
>     ) ‑> Optional[Dict[str, Any]]


Predict the new metadata syntax based on the raw metadata.


Returns
-----=
<code>Optional\[Dict\[str, Any]]</code>
:   The predicted new metadata syntax, or None if prediction fails.





    
# Module `metagpt.predictors.predictor_network` {#id}

This module contains the PredictorNetwork class, which uses a network of predictors
to process, annotate, and merge metadata for microscopy images.





    
## Classes


    
### Class `PredictorNetwork` {#id}




>     class PredictorNetwork(
>         raw_meta: str
>     )


A predictor class that uses a network of predictors to process and annotate metadata.

This predictor approach uses three assistants:
1. A separator to split the raw metadata into already contained and new metadata.
2. An annotator to predict structured annotations from the new metadata.
3. A simple predictor to process the remaining metadata.

Initialize the PredictorNetwork.


Args
-----=
**```raw_meta```** :&ensp;<code>str</code>
:   The raw metadata to be processed and annotated.




    
#### Ancestors (in MRO)

* [metagpt.predictors.predictor_template.PredictorTemplate](#metagpt.predictors.predictor_template.PredictorTemplate)






    
#### Methods


    
##### Method `predict` {#id}




>     def predict(
>         self
>     ) ‑> Tuple[Optional[str], float, int]


Predict structured annotations based on the raw metadata.

This method uses three predictors in sequence:
1. PredictorSeperator to split the metadata.
2. PredictorSimpleAnnotation to generate annotations.
3. PredictorSimple to process the remaining metadata.


Returns
-----=
<code>Tuple\[Optional\[str], float, int]</code>
:   
    - The merged XML annotation (or None if prediction fails)
    - The total cost of the prediction
    - The total number of attempts made





    
# Module `metagpt.predictors.predictor_network_annotator` {#id}

This module contains the PredictorNetworkAnnotation class, which uses a network
of predictors to process and annotate metadata for microscopy images.





    
## Classes


    
### Class `PredictorNetworkAnnotation` {#id}




>     class PredictorNetworkAnnotation(
>         raw_meta: str
>     )


A predictor class that uses two assistants to process and annotate metadata.

This predictor approach uses two assistants:
1. A separator to split the raw metadata into already contained and new metadata.
2. An annotator to predict structured annotations from the new metadata.

Initialize the PredictorNetworkAnnotation.


Args
-----=
**```raw_meta```** :&ensp;<code>str</code>
:   The raw metadata to be processed and annotated.




    
#### Ancestors (in MRO)

* [metagpt.predictors.predictor_template.PredictorTemplate](#metagpt.predictors.predictor_template.PredictorTemplate)






    
#### Methods


    
##### Method `predict` {#id}




>     def predict(
>         self
>     ) ‑> Tuple[Optional[Any], float, float]


Predict structured annotations based on the raw metadata.

This method uses two predictors in sequence:
1. PredictorSeperator to split the metadata.
2. PredictorSimpleAnnotation to generate annotations.


Returns
-----=
<code>Tuple\[Optional\[Any], float, float]</code>
:   
    - The predicted annotations (or None if prediction fails)
    - The total cost of the prediction
    - The total number of attempts made





    
# Module `metagpt.predictors.predictor_seperator` {#id}

This module contains the PredictorSeperator class, which is responsible for
separating raw metadata into structured annotations and OME properties.





    
## Classes


    
### Class `PredictorSeperator` {#id}




>     class PredictorSeperator(
>         raw_meta: str
>     )


A predictor class that separates raw metadata into structured annotations
and OME properties using OpenAI's language model and vector embeddings.

Initialize the PredictorSeperator.


Args
-----=
**```raw_meta```** :&ensp;<code>str</code>
:   The raw metadata to be processed.




    
#### Ancestors (in MRO)

* [metagpt.predictors.predictor_template.PredictorTemplate](#metagpt.predictors.predictor_template.PredictorTemplate)



    
#### Class variables


    
##### Variable `SepOutputTool` {#id}




This tool automatically formats and structures the metadata in the appropriate way.




    
#### Methods


    
##### Method `init_assistant` {#id}




>     def init_assistant(
>         self
>     ) ‑> None


Initialize the OpenAI assistant.

    
##### Method `init_run` {#id}




>     def init_run(
>         self
>     ) ‑> None


Initialize and monitor the run of the assistant.

    
##### Method `predict` {#id}




>     def predict(
>         self
>     ) ‑> Tuple[Optional[Tuple[Dict[str, str], Dict[str, str]]], float, int]


Predict the separation of raw metadata into structured annotations and OME properties.


Returns
-----=
Tuple[Optional[Tuple[Dict[str, str], Dict[str, str]]], float, int]:
    - A tuple containing two dictionaries (annotation_properties, ome_properties),
      or None if prediction fails
    - The cost of the prediction
    - The number of attempts made



    
# Module `metagpt.predictors.predictor_simple` {#id}

This module contains the PredictorSimple class, which is responsible for
predicting well-formed OME XML from raw metadata using OpenAI's language model.





    
## Classes


    
### Class `PredictorSimple` {#id}




>     class PredictorSimple(
>         raw_meta: str
>     )


A predictor class that generates well-formed OME XML from raw metadata
using OpenAI's language model and vector embeddings.

Initialize the PredictorSimple.


Args
-----=
**```raw_meta```** :&ensp;<code>str</code>
:   The raw metadata to be processed.




    
#### Ancestors (in MRO)

* [metagpt.predictors.predictor_template.PredictorTemplate](#metagpt.predictors.predictor_template.PredictorTemplate)



    
#### Class variables


    
##### Variable `OMEXMLResponse` {#id}




The response containing the well-formed OME XML.




    
#### Methods


    
##### Method `init_assistant` {#id}




>     def init_assistant(
>         self
>     ) ‑> None


Initialize the OpenAI assistant.

    
##### Method `init_run` {#id}




>     def init_run(
>         self
>     ) ‑> None


Initialize and monitor the run of the assistant.

    
##### Method `predict` {#id}




>     def predict(
>         self
>     ) ‑> Tuple[Optional[str], float, int]


Predict well-formed OME XML based on the raw metadata.


Returns
-----=
Tuple[Optional[str], float, int]:
    - The predicted OME XML as a string, or None if prediction fails
    - The cost of the prediction
    - The number of attempts made



    
# Module `metagpt.predictors.predictor_simple_annotator` {#id}

This module contains the PredictorSimpleAnnotation class, which is responsible for
predicting structured annotations for the OME model from raw metadata.





    
## Classes


    
### Class `PredictorSimpleAnnotation` {#id}




>     class PredictorSimpleAnnotation(
>         raw_meta: str
>     )


A predictor class that generates structured annotations for the OME model
from raw metadata using OpenAI's language model.

Initialize the PredictorSimpleAnnotation.


Args
-----=
**```raw_meta```** :&ensp;<code>str</code>
:   The raw metadata to be processed.




    
#### Ancestors (in MRO)

* [metagpt.predictors.predictor_template.PredictorTemplate](#metagpt.predictors.predictor_template.PredictorTemplate)



    
#### Class variables


    
##### Variable `XMLAnnotationFunction` {#id}




The function call to hand in the structured annotations to the OME XML.




    
#### Methods


    
##### Method `init_assistant` {#id}




>     def init_assistant(
>         self
>     ) ‑> None


Initialize the OpenAI assistant.

    
##### Method `init_run` {#id}




>     def init_run(
>         self
>     ) ‑> None


Initialize and monitor the run of the assistant.

    
##### Method `predict` {#id}




>     def predict(
>         self
>     ) ‑> Tuple[Optional[Dict[str, Any]], float, int]


Predict structured annotations based on the raw metadata.


Returns
-----=
Tuple[Optional[Dict[str, Any]], float, int]:
    - The predicted annotations as a dictionary, or None if prediction fails
    - The cost of the prediction
    - The number of attempts made



    
# Module `metagpt.predictors.predictor_state` {#id}

This module contains the PredictorState class, which is responsible for
predicting and updating OME metadata using JSON patches and OpenAI's language model.





    
## Classes


    
### Class `AddReplaceTestOperation` {#id}




>     class AddReplaceTestOperation(
>         **data: Any
>     )


Model for Add, Replace, and Test operations in JSON Patch.

Create a new model by parsing and validating input data from keyword arguments.

Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

<code>self</code> is explicitly positional-only to allow <code>self</code> as a field name.


    
#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)



    
#### Class variables


    
##### Variable `model_computed_fields` {#id}






    
##### Variable `model_config` {#id}






    
##### Variable `model_fields` {#id}






    
##### Variable `op` {#id}



Type: `Literal['add', 'replace', 'test']`



    
##### Variable `path` {#id}



Type: `str`



    
##### Variable `value` {#id}



Type: `Any`






    
### Class `JsonPatch` {#id}




>     class JsonPatch(
>         **data: Any
>     )


Model for a complete JSON Patch.

Create a new model by parsing and validating input data from keyword arguments.

Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

<code>self</code> is explicitly positional-only to allow <code>self</code> as a field name.


    
#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)



    
#### Class variables


    
##### Variable `Config` {#id}






    
##### Variable `model_computed_fields` {#id}






    
##### Variable `model_config` {#id}






    
##### Variable `model_fields` {#id}






    
##### Variable `root` {#id}



Type: `List[Union[metagpt.predictors.predictor_state.AddReplaceTestOperation, metagpt.predictors.predictor_state.RemoveOperation, metagpt.predictors.predictor_state.MoveCopyOperation]]`






    
### Class `MoveCopyOperation` {#id}




>     class MoveCopyOperation(
>         **data: Any
>     )


Model for Move and Copy operations in JSON Patch.

Create a new model by parsing and validating input data from keyword arguments.

Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

<code>self</code> is explicitly positional-only to allow <code>self</code> as a field name.


    
#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)



    
#### Class variables


    
##### Variable `from_` {#id}



Type: `str`



    
##### Variable `model_computed_fields` {#id}






    
##### Variable `model_config` {#id}






    
##### Variable `model_fields` {#id}






    
##### Variable `op` {#id}



Type: `Literal['move', 'copy']`



    
##### Variable `path` {#id}



Type: `str`






    
### Class `PredictorState` {#id}




>     class PredictorState(
>         raw_meta: str,
>         state: pydantic.main.BaseModel = None
>     )


A predictor class that generates and applies JSON patches to update OME metadata
using OpenAI's language model.

Initialize the PredictorState.


Args
-----=
**```raw_meta```** :&ensp;<code>str</code>
:   The raw metadata to process.


**```state```** :&ensp;<code>BaseModel</code>, optional
:   The initial state. Defaults to OME().




    
#### Ancestors (in MRO)

* [metagpt.predictors.predictor_template.PredictorTemplate](#metagpt.predictors.predictor_template.PredictorTemplate)






    
#### Methods


    
##### Method `init_assistant` {#id}




>     def init_assistant(
>         self
>     ) ‑> None


Initialize the OpenAI assistant.

    
##### Method `init_run` {#id}




>     def init_run(
>         self
>     ) ‑> None


Initialize and monitor the run of the assistant.

    
##### Method `predict` {#id}




>     def predict(
>         self,
>         indent: Optional[int] = 0
>     ) ‑> tuple[typing.Optional[str], float, int]


Predict OME metadata and apply JSON patches to update the state.


Args
-----=
**```indent```** :&ensp;<code>Optional\[int]</code>
:   Indentation for logging. Defaults to 0.



Returns
-----=
<code>tuple\[Optional\[str], float, int]</code>
:   The updated OME XML, cost, and number of attempts.



    
### Class `RemoveOperation` {#id}




>     class RemoveOperation(
>         **data: Any
>     )


Model for Remove operation in JSON Patch.

Create a new model by parsing and validating input data from keyword arguments.

Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

<code>self</code> is explicitly positional-only to allow <code>self</code> as a field name.


    
#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)



    
#### Class variables


    
##### Variable `model_computed_fields` {#id}






    
##### Variable `model_config` {#id}






    
##### Variable `model_fields` {#id}






    
##### Variable `op` {#id}



Type: `Literal['remove']`



    
##### Variable `path` {#id}



Type: `str`






    
### Class `update_json_state` {#id}




>     class update_json_state(
>         **data: Any
>     )


Model for updating the state of the predictor from a list of JSON patches.

Create a new model by parsing and validating input data from keyword arguments.

Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

<code>self</code> is explicitly positional-only to allow <code>self</code> as a field name.


    
#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)



    
#### Class variables


    
##### Variable `json_patches` {#id}



Type: `Optional[List[metagpt.predictors.predictor_state.JsonPatch]]`



    
##### Variable `model_computed_fields` {#id}






    
##### Variable `model_config` {#id}






    
##### Variable `model_fields` {#id}











    
# Module `metagpt.predictors.predictor_state_tree` {#id}

This module contains the PredictorStateTree class and related components for
predicting OME metadata using a tree-based approach with OpenAI's language model.




    
## Functions


    
### Function `create_instance` {#id}




>     def create_instance(
>         instance: Type[pydantic.main.BaseModel],
>         obj_dict: Dict[str, Any]
>     ) ‑> Optional[pydantic.main.BaseModel]


Create an instance of a Pydantic model, filling it with child objects.


Args
-----=
**```instance```** :&ensp;<code>Type\[BaseModel]</code>
:   The Pydantic model class to instantiate.


**```obj_dict```** :&ensp;<code>Dict\[str, Any]</code>
:   Dictionary of child objects to include.



Returns
-----=
<code>Optional\[BaseModel]</code>
:   The instantiated model, or None if instantiation fails.




    
## Classes


    
### Class `PredictorStateTree` {#id}




>     class PredictorStateTree(
>         raw_meta: str,
>         model: Type[pydantic.main.BaseModel] = None
>     )


A predictor class that uses a tree-based approach to predict OME metadata.

Initialize the PredictorStateTree.


Args
-----=
**```raw_meta```** :&ensp;<code>str</code>
:   The raw metadata to process.


**```model```** :&ensp;<code>Type\[BaseModel]</code>, optional
:   The root model to use. Defaults to OME.




    
#### Ancestors (in MRO)

* [metagpt.predictors.predictor_template.PredictorTemplate](#metagpt.predictors.predictor_template.PredictorTemplate)






    
#### Methods


    
##### Method `build_tree` {#id}




>     def build_tree(
>         self,
>         root_model: Type[pydantic.main.BaseModel]
>     ) ‑> metagpt.predictors.predictor_state_tree.TreeNode


Build the complete dependency tree starting from the root model.


Args
-----=
**```root_model```** :&ensp;<code>Type\[BaseModel]</code>
:   The root model to start building the tree from.



Returns
-----=
<code>[TreeNode](#metagpt.predictors.predictor\_state\_tree.TreeNode "metagpt.predictors.predictor\_state\_tree.TreeNode")</code>
:   The root node of the built tree.



    
##### Method `collect_dependencies` {#id}




>     def collect_dependencies(
>         self,
>         model: Type[pydantic.main.BaseModel],
>         known_models: Dict[str, Type[pydantic.main.BaseModel]],
>         collected: Dict[str, Type[pydantic.main.BaseModel]]
>     ) ‑> None


Collect all dependent models for a given model.


Args
-----=
**```model```** :&ensp;<code>Type\[BaseModel]</code>
:   The model to collect dependencies for.


**```known_models```** :&ensp;<code>Dict\[str, Type\[BaseModel]]</code>
:   Dictionary of known models.


**```collected```** :&ensp;<code>Dict\[str, Type\[BaseModel]]</code>
:   Dictionary to store collected models.



    
##### Method `create_dependency_tree` {#id}




>     def create_dependency_tree(
>         self,
>         model: Type[pydantic.main.BaseModel],
>         known_models: Dict[str, Type[pydantic.main.BaseModel]],
>         visited: Set[str]
>     ) ‑> metagpt.predictors.predictor_state_tree.TreeNode


Create a dependency tree for a given model.


Args
-----=
**```model```** :&ensp;<code>Type\[BaseModel]</code>
:   The model to create a tree for.


**```known_models```** :&ensp;<code>Dict\[str, Type\[BaseModel]]</code>
:   Dictionary of known models.


**```visited```** :&ensp;<code>Set\[str]</code>
:   Set of visited model names.



Returns
-----=
<code>[TreeNode](#metagpt.predictors.predictor\_state\_tree.TreeNode "metagpt.predictors.predictor\_state\_tree.TreeNode")</code>
:   The root node of the created tree.



    
##### Method `predict` {#id}




>     def predict(
>         self
>     ) ‑> Tuple[Optional[pydantic.main.BaseModel], Optional[float], Optional[int]]


Predict the OME metadata using the dependency tree.


Returns
-----=
Tuple[Optional[BaseModel], Optional[float], Optional[int]]:
    The predicted metadata, cost (None for this implementation), and attempts (None for this implementation).

    
##### Method `print_tree` {#id}




>     def print_tree(
>         self,
>         node: Optional[metagpt.predictors.predictor_state_tree.TreeNode] = None,
>         indent: str = ''
>     ) ‑> None


Print the structure of the dependency tree.


Args
-----=
**```node```** :&ensp;<code>Optional\[[TreeNode](#metagpt.predictors.predictor\_state\_tree.TreeNode "metagpt.predictors.predictor\_state\_tree.TreeNode")]</code>
:   The node to start printing from. If None, starts from the root.


**```indent```** :&ensp;<code>str</code>
:   The current indentation string.



    
### Class `TreeNode` {#id}




>     class TreeNode(
>         model: Type[pydantic.main.BaseModel]
>     )


Represents a node in the dependency tree for OME metadata prediction.







    
#### Methods


    
##### Method `add_child` {#id}




>     def add_child(
>         self,
>         child: TreeNode
>     ) ‑> None


Add a child node to this node.

    
##### Method `instantiate_model` {#id}




>     def instantiate_model(
>         self,
>         child_objects: Dict[str, Any]
>     ) ‑> pydantic.main.BaseModel


Instantiate the model for this node, including child objects.


Args
-----=
**```child_objects```** :&ensp;<code>Dict\[str, Any]</code>
:   Dictionary of child objects to include.



Returns
-----=
<code>BaseModel</code>
:   The instantiated model, or a MaybeModel if instantiation fails.



    
##### Method `predict_meta` {#id}




>     def predict_meta(
>         self,
>         raw_meta: str,
>         indent: int = 0
>     ) ‑> Tuple[Optional[pydantic.main.BaseModel], float, int]


Predict metadata for this node and its children.


Args
-----=
**```raw_meta```** :&ensp;<code>str</code>
:   The raw metadata to process.


**```indent```** :&ensp;<code>int</code>
:   The indentation level for printing (used for debugging).



Returns
-----=
<code>Tuple\[Optional\[BaseModel], float, int]</code>
:   The predicted state, total cost, and total attempts.



    
##### Method `required_fields` {#id}




>     def required_fields(
>         self,
>         model: Type[pydantic.main.BaseModel],
>         recursive: bool = False
>     ) ‑> collections.abc.Iterator[str]


Get all required fields of a Pydantic model, optionally including nested models.


Args
-----=
**```model```** :&ensp;<code>Type\[BaseModel]</code>
:   The Pydantic model to inspect.


**```recursive```** :&ensp;<code>bool</code>
:   Whether to include fields from nested models.



Yields
-----=
<code>str</code>
:   Names of required fields.





    
# Module `metagpt.predictors.predictor_template` {#id}

This module contains the PredictorTemplate class, which serves as a base class
for creating predictors that utilize OpenAI's API to generate OME XML from raw metadata.





    
## Classes


    
### Class `PredictorTemplate` {#id}




>     class PredictorTemplate


A template for creating a new predictor. A predictor utilizes one or several assistants
to predict the OME XML from the raw metadata.



    
#### Descendants

* [metagpt.predictors.predictor_distorter.PredictorDistorter](#metagpt.predictors.predictor_distorter.PredictorDistorter)
* [metagpt.predictors.predictor_network.PredictorNetwork](#metagpt.predictors.predictor_network.PredictorNetwork)
* [metagpt.predictors.predictor_network_annotator.PredictorNetworkAnnotation](#metagpt.predictors.predictor_network_annotator.PredictorNetworkAnnotation)
* [metagpt.predictors.predictor_seperator.PredictorSeperator](#metagpt.predictors.predictor_seperator.PredictorSeperator)
* [metagpt.predictors.predictor_simple.PredictorSimple](#metagpt.predictors.predictor_simple.PredictorSimple)
* [metagpt.predictors.predictor_simple_annotator.PredictorSimpleAnnotation](#metagpt.predictors.predictor_simple_annotator.PredictorSimpleAnnotation)
* [metagpt.predictors.predictor_state.PredictorState](#metagpt.predictors.predictor_state.PredictorState)
* [metagpt.predictors.predictor_state_tree.PredictorStateTree](#metagpt.predictors.predictor_state_tree.PredictorStateTree)





    
#### Methods


    
##### Method `add_attempts` {#id}




>     def add_attempts(
>         self,
>         i: float = 1
>     ) ‑> None


Add an attempt to the attempt counter. Normalized by the number of assistants.


Args
-----=
**```i```** :&ensp;<code>float</code>
:   The number of attempts to add. Defaults to 1.



    
##### Method `clean_assistants` {#id}




>     def clean_assistants(
>         self
>     ) ‑> None


Clean up the assistants, threads, and vector stores.

    
##### Method `export_ome_xml` {#id}




>     def export_ome_xml(
>         self
>     ) ‑> None


Export the OME XML response to a file.

    
##### Method `generate_message` {#id}




>     def generate_message(
>         self,
>         msg: Optional[str] = None
>     ) ‑> Any


Generate a new message in the thread.


Args
-----=
**```msg```** :&ensp;<code>Optional\[str]</code>
:   The message content. If None, uses self.message.



Returns
-----=
<code>Any</code>
:   The created message object.



    
##### Method `get_cost` {#id}




>     def get_cost(
>         self
>     ) ‑> Optional[float]


Calculate the cost of the prediction based on token usage.


Returns
-----=
<code>Optional\[float]</code>
:   The calculated cost, or None if there was an error.



    
##### Method `get_response` {#id}




>     def get_response(
>         self
>     ) ‑> None


Predict the OME XML from the raw metadata and handle the response.

    
##### Method `init_thread` {#id}




>     def init_thread(
>         self
>     ) ‑> None


Initialize a new thread with the initial prompt and message.

    
##### Method `init_vector_store` {#id}




>     def init_vector_store(
>         self
>     ) ‑> None


Initialize the vector store and upload file batches.

    
##### Method `predict` {#id}




>     def predict(
>         self
>     ) ‑> Dict[str, Any]


Predict the OME XML from the raw metadata.


Returns
-----=
<code>Dict\[str, Any]</code>
:   The predicted OME XML and related information.



Raises
-----=
<code>NotImplementedError</code>
:   This method should be implemented by subclasses.



    
##### Method `read_ome_as_string` {#id}




>     def read_ome_as_string(
>         self,
>         path: str
>     ) ‑> str


Read the OME XML as a string from a file.


Args
-----=
**```path```** :&ensp;<code>str</code>
:   The path to the OME XML file.



Returns
-----=
<code>str</code>
:   The contents of the OME XML file as a string.



    
##### Method `read_ome_as_xml` {#id}




>     def read_ome_as_xml(
>         self,
>         path: str
>     ) ‑> str


Read the OME XML file and return the root element as a string.


Args
-----=
**```path```** :&ensp;<code>str</code>
:   The path to the OME XML file.



Returns
-----=
<code>str</code>
:   The root element of the OME XML as a string.



    
##### Method `read_raw_metadata` {#id}




>     def read_raw_metadata(
>         self
>     ) ‑> str


Read the raw metadata from the file.


Returns
-----=
<code>str</code>
:   The contents of the raw metadata file.



Raises
-----=
<code>FileNotFoundError</code>
:   If the path_to_raw_metadata is not set or the file doesn't exist.



    
##### Method `subdivide_raw_metadata` {#id}




>     def subdivide_raw_metadata(
>         self
>     ) ‑> None


Subdivide the raw metadata into appropriate chunks.

    
##### Method `validate` {#id}




>     def validate(
>         self,
>         ome_xml: str
>     ) ‑> Optional[Exception]


Validate the OME XML against the OME XSD.


Args
-----=
**```ome_xml```** :&ensp;<code>str</code>
:   The OME XML string to validate.



Returns
-----=
<code>Optional\[Exception]</code>
:   The exception if validation fails, None otherwise.





    
# Module `metagpt.utils` {#id}




    
## Sub-modules

* [metagpt.utils.BioformatsReader](#metagpt.utils.BioformatsReader)
* [metagpt.utils.DataClasses](#metagpt.utils.DataClasses)
* [metagpt.utils.utils](#metagpt.utils.utils)






    
# Module `metagpt.utils.BioformatsReader` {#id}

This module implements functions to read proprietary images and return their metadata
in OME-XML format and as raw metadata key-value pairs using Bio-Formats.




    
## Functions


    
### Function `get_omexml_metadata` {#id}




>     def get_omexml_metadata(
>         path: Optional[str] = None,
>         url: Optional[str] = None
>     ) ‑> str


Read the OME metadata from a file using Bio-Formats.


Args
-----=
**```path```** :&ensp;<code>Optional\[str]</code>
:   Path to the file. Defaults to None.


**```url```** :&ensp;<code>Optional\[str]</code>
:   URL of the file. Defaults to None.



Returns
-----=
<code>str</code>
:   The metadata as XML.



Raises
-----=
<code>ValueError</code>
:   If neither path nor url is provided.



    
### Function `get_raw_metadata` {#id}




>     def get_raw_metadata(
>         path: str
>     ) ‑> Dict[str, str]


Read the raw metadata from a file using Bio-Formats.


Args
-----=
**```path```** :&ensp;<code>str</code>
:   Path to the file.



Returns
-----=
<code>Dict\[str, str]</code>
:   The metadata as a dictionary.



    
### Function `raw_to_tree` {#id}




>     def raw_to_tree(
>         raw_metadata: Dict[str, str]
>     ) ‑> Dict[str, Union[str, Dict]]


Convert the raw metadata to a tree structure by separating the key on the "|" character.


Args
-----=
**```raw_metadata```** :&ensp;<code>Dict\[str, str]</code>
:   The raw metadata dictionary.



Returns
-----=
<code>Dict\[str, Union\[str, Dict]]</code>
:   The metadata in a tree structure.






    
# Module `metagpt.utils.DataClasses` {#id}

Data classes for the metagpt package.





    
## Classes


    
### Class `Dataset` {#id}




>     class Dataset(
>         name: str = None,
>         samples: dict[slice(<class 'str'>, <class 'metagpt.utils.DataClasses.Sample'>, None)] = FieldInfo(annotation=NoneType, required=False, default_factory=dict),
>         cost: Optional[float] = 0,
>         time: Optional[float] = 0
>     )


Dataset(name: str = None, samples: dict[slice(<class 'str'>, <class 'metagpt.utils.DataClasses.Sample'>, None)] = FieldInfo(annotation=NoneType, required=False, default_factory=dict), cost: Optional[float] = 0, time: Optional[float] = 0)




    
#### Class variables


    
##### Variable `cost` {#id}



Type: `Optional[float]`



    
##### Variable `name` {#id}



Type: `str`



    
##### Variable `samples` {#id}



Type: `dict[slice(<class 'str'>, <class 'metagpt.utils.DataClasses.Sample'>, None)]`



    
##### Variable `time` {#id}



Type: `Optional[float]`






    
#### Methods


    
##### Method `add_sample` {#id}




>     def add_sample(
>         self,
>         sample: metagpt.utils.DataClasses.Sample
>     )




    
### Class `Sample` {#id}




>     class Sample(
>         format: str,
>         attempts: float,
>         index: int,
>         file_name: str,
>         name: str = None,
>         metadata_str: str = None,
>         method: str = None,
>         metadata_xml: ome_types._autogenerated.ome_2016_06.ome.OME = FieldInfo(annotation=NoneType, required=False, default_factory=OME, description='The metadata as an OME object'),
>         cost: Optional[float] = None,
>         paths: Optional[list[str]] = None,
>         time: Optional[float] = None,
>         gpt_model: Optional[str] = None
>     )


Sample(format: str, attempts: float, index: int, file_name: str, name: str = None, metadata_str: str = None, method: str = None, metadata_xml: ome_types._autogenerated.ome_2016_06.ome.OME = FieldInfo(annotation=NoneType, required=False, default_factory=OME, description='The metadata as an OME object'), cost: Optional[float] = None, paths: Optional[list[str]] = None, time: Optional[float] = None, gpt_model: Optional[str] = None)




    
#### Class variables


    
##### Variable `attempts` {#id}



Type: `float`



    
##### Variable `cost` {#id}



Type: `Optional[float]`



    
##### Variable `file_name` {#id}



Type: `str`



    
##### Variable `format` {#id}



Type: `str`



    
##### Variable `gpt_model` {#id}



Type: `Optional[str]`



    
##### Variable `index` {#id}



Type: `int`



    
##### Variable `metadata_str` {#id}



Type: `str`



    
##### Variable `metadata_xml` {#id}



Type: `ome_types._autogenerated.ome_2016_06.ome.OME`



    
##### Variable `method` {#id}



Type: `str`



    
##### Variable `name` {#id}



Type: `str`



    
##### Variable `paths` {#id}



Type: `Optional[list[str]]`



    
##### Variable `time` {#id}



Type: `Optional[float]`








    
# Module `metagpt.utils.utils` {#id}

This module contains various utility functions and classes for handling OME XML data,
JSON operations, and other helper functions used in the MetaGPT project.




    
## Functions


    
### Function `browse_schema` {#id}




>     def browse_schema(
>         cls: Type[pydantic.main.BaseModel],
>         additional_ignored_keywords: List[str] = [],
>         max_depth: int = inf
>     ) ‑> Dict[str, Any]


Browse a schema as jsonschema, with depth control.


Args
-----=
**```cls```** :&ensp;<code>Type\[BaseModel]</code>
:   The Pydantic model to convert to a schema.


**```additional_ignored_keywords```** :&ensp;<code>List\[str]</code>, optional
:   Additional keywords to ignore in the schema. Defaults to [].


**```max_depth```** :&ensp;<code>int</code>, optional
:   Maximum depth of nesting to include in the schema. Defaults to infinity.



Returns
-----=
<code>Dict\[str, Any]</code>
:   A dictionary in the format of OpenAI's schema as jsonschema.



    
### Function `camel_to_snake` {#id}




>     def camel_to_snake(
>         name: str
>     ) ‑> str


Convert a CamelCase string to snake_case.


Args
-----=
**```name```** :&ensp;<code>str</code>
:   The CamelCase string to convert.



Returns
-----=
<code>str</code>
:   The converted snake_case string.



    
### Function `custom_apply` {#id}




>     def custom_apply(
>         patch: jsonpatch.JsonPatch,
>         data: Dict[str, Any]
>     ) ‑> Dict[str, Any]


Apply the JSON Patch, automatically creating missing nodes.


Args
-----=
**```patch```** :&ensp;<code>jsonpatch.JsonPatch</code>
:   The JSON Patch to apply.


**```data```** :&ensp;<code>Dict\[str, Any]</code>
:   The data to apply the patch to.



Returns
-----=
<code>Dict\[str, Any]</code>
:   The updated data after applying the patch.



    
### Function `dict_to_xml_annotation` {#id}




>     def dict_to_xml_annotation(
>         value: Dict[str, Any]
>     ) ‑> ome_types._autogenerated.ome_2016_06.xml_annotation.XMLAnnotation


Convert a dictionary to an XMLAnnotation object, handling nested dictionaries.


Args
-----=
**```value```** :&ensp;<code>Dict\[str, Any]</code>
:   The dictionary to be converted to an XMLAnnotation object. 
    It requires the key 'annotations' which is a dictionary of key-value pairs.



Returns
-----=
<code>XMLAnnotation</code>
:   The resulting XMLAnnotation object.



    
### Function `ensure_path_exists` {#id}




>     def ensure_path_exists(
>         data: Dict[str, Any],
>         path: str
>     ) ‑> None


Ensure that the path exists in the data structure, creating empty lists or dicts as needed.


Args
-----=
**```data```** :&ensp;<code>Dict\[str, Any]</code>
:   The data structure to modify.


**```path```** :&ensp;<code>str</code>
:   The path to ensure exists.



    
### Function `flatten` {#id}




>     def flatten(
>         container: Union[List, Tuple, Set]
>     ) ‑> Generator[Any, None, None]


Flatten a nested container (list, tuple, or set).


Args
-----=
**```container```** :&ensp;<code>Union\[List, Tuple, Set]</code>
:   The nested container to flatten.



Yields
-----=
<code>Any</code>
:   Each non-container element in the flattened structure.



    
### Function `from_dict` {#id}




>     def from_dict(
>         ome_dict: Dict[str, Any],
>         state: Optional[ome_types._autogenerated.ome_2016_06.ome.OME] = None
>     ) ‑> ome_types._autogenerated.ome_2016_06.ome.OME


Convert a dictionary to an OME object.


Args
-----=
**```ome_dict```** :&ensp;<code>Dict\[str, Any]</code>
:   The dictionary to convert.


**```state```** :&ensp;<code>Optional\[OME]</code>
:   The initial OME state to update.



Returns
-----=
<code>OME</code>
:   The resulting OME object.



    
### Function `generate_paths` {#id}




>     def generate_paths(
>         json_data: Union[Dict[str, Any], List[Any]],
>         current_path: str = '',
>         paths: List[str] = None
>     ) ‑> List[str]


Generate all possible paths from a nested JSON structure.


Args
-----=
**```json_data```** :&ensp;<code>Union\[Dict\[str, Any], List\[Any]]</code>
:   The nested JSON structure to traverse.


**```current_path```** :&ensp;<code>str</code>, optional
:   The current path being built. Defaults to "".


**```paths```** :&ensp;<code>List\[str]</code>, optional
:   The list to store all generated paths. Defaults to None.



Returns
-----=
<code>List\[str]</code>
:   A list of strings, where each string represents a path in the format "path/to/element = value".



    
### Function `get_json` {#id}




>     def get_json(
>         xml_root: xml.etree.ElementTree.Element,
>         paths: Dict[str, Any] = {}
>     ) ‑> Dict[str, Any]


Convert an XML tree to a JSON-like dictionary structure.


Args
-----=
**```xml_root```** :&ensp;<code>ET.Element</code>
:   The root element of the XML tree.


**```paths```** :&ensp;<code>Dict\[str, Any]</code>, optional
:   A dictionary to store the converted structure. Defaults to {}.



Returns
-----=
<code>Dict\[str, Any]</code>
:   The JSON-like dictionary representation of the XML tree.



    
### Function `load_output` {#id}




>     def load_output(
>         path: str
>     ) ‑> Tuple[Optional[str], Optional[float], Optional[float], Optional[float]]


Load output from a file.


Args
-----=
**```path```** :&ensp;<code>str</code>
:   The file path to load from.



Returns
-----=
<code>Tuple\[Optional\[str], Optional\[float], Optional\[float], Optional\[float]]</code>
:   
    The loaded output, cost, attempts, and prediction time.



    
### Function `make_prediction` {#id}




>     def make_prediction(
>         predictor: Any,
>         in_data: Any,
>         dataset: metagpt.utils.DataClasses.Dataset,
>         file_name: str,
>         index: int,
>         should_predict: str = 'maybe',
>         start_point: Optional[str] = None,
>         data_format: Optional[str] = None,
>         model: Optional[str] = None,
>         out_path: Optional[str] = None
>     ) ‑> None


Make a prediction using the specified predictor and add the result to the dataset.


Args
-----=
**```predictor```** :&ensp;<code>Any</code>
:   The predictor object.


**```in_data```** :&ensp;<code>Any</code>
:   Input data for the prediction.


**```dataset```** :&ensp;<code>Dataset</code>
:   The dataset to add the prediction to.


**```file_name```** :&ensp;<code>str</code>
:   Name of the file being processed.


**```index```** :&ensp;<code>int</code>
:   Index of the prediction.


**```should_predict```** :&ensp;<code>str</code>, optional
:   Whether to predict. Defaults to "maybe".


**```start_point```** :&ensp;<code>Optional\[str]</code>, optional
:   Starting point for the prediction. Defaults to None.


**```data_format```** :&ensp;<code>Optional\[str]</code>, optional
:   Data format. Defaults to None.


**```model```** :&ensp;<code>Optional\[str]</code>, optional
:   Model to use for prediction. Defaults to None.


**```out_path```** :&ensp;<code>Optional\[str]</code>, optional
:   Output path. Defaults to None.



    
### Function `merge_xml_annotation` {#id}




>     def merge_xml_annotation(
>         annot: Dict[str, Any],
>         ome: Optional[str] = None
>     ) ‑> Optional[str]


Merge the annotation section with the OME XML.


Args
-----=
**```ome```** :&ensp;<code>Optional\[str]</code>
:   The OME XML string.


**```annot```** :&ensp;<code>Dict\[str, Any]</code>
:   The annotation dictionary.



Returns
-----=
<code>Optional\[str]</code>
:   The merged XML string, or None if inputs are invalid.



    
### Function `num_tokens_from_string` {#id}




>     def num_tokens_from_string(
>         string: str,
>         encoding_name: str = 'cl100k_base'
>     ) ‑> int


Returns the number of tokens in a text string.


Args
-----=
**```string```** :&ensp;<code>str</code>
:   The input string to tokenize.


**```encoding_name```** :&ensp;<code>str</code>, optional
:   The name of the tokenizer encoding to use. Defaults to "cl100k_base".



Returns
-----=
<code>int</code>
:   The number of tokens in the input string.



    
### Function `read_ome_xml` {#id}




>     def read_ome_xml(
>         path: str
>     ) ‑> xml.etree.ElementTree.Element


Read an OME XML file and return the root element.


Args
-----=
**```path```** :&ensp;<code>str</code>
:   The path to the OME XML file.



Returns
-----=
<code>ET.Element</code>
:   The root element of the XML tree.



    
### Function `render_cell_output` {#id}




>     def render_cell_output(
>         output_path: str
>     ) ‑> None


Load the captured output from a file and render it.


Args
-----=
**```output_path```** :&ensp;<code>str</code>
:   Path to the output file where the cell output is saved.



    
### Function `safe_float` {#id}




>     def safe_float(
>         value: Any
>     ) ‑> Optional[float]


Safely convert a value to float, returning None if conversion is not possible.


Args
-----=
**```value```** :&ensp;<code>Any</code>
:   The value to convert to float.



Returns
-----=
<code>Optional\[float]</code>
:   The float value if conversion is successful, None otherwise.



    
### Function `save_and_stream_output` {#id}




>     def save_and_stream_output(
>         output_path: str = 'out/jupyter_cell_outputs/cell_output_2024-07-27T19:37:02.981914_.json'
>     )


Context manager to capture the output of a code block, save it to a file,
and print it to the console in real-time.


Args
-----=
**```output_path```** :&ensp;<code>str</code>
:   Path to the output file where the cell output will be saved.



    
### Function `save_output` {#id}




>     def save_output(
>         output: str,
>         cost: float,
>         attempts: float,
>         pred_time: float,
>         path: str
>     ) ‑> bool


Save output to a file.


Args
-----=
**```output```** :&ensp;<code>str</code>
:   The output to save.


**```cost```** :&ensp;<code>float</code>
:   The cost of the prediction.


**```attempts```** :&ensp;<code>float</code>
:   The number of attempts made.


**```pred_time```** :&ensp;<code>float</code>
:   The prediction time.


**```path```** :&ensp;<code>str</code>
:   The file path to save to.



Returns
-----=
<code>bool</code>
:   True if save was successful, False otherwise.



    
### Function `update_state` {#id}




>     def update_state(
>         current_state: ome_types._autogenerated.ome_2016_06.ome.OME,
>         proposed_change: List[Dict[str, Any]]
>     ) ‑> ome_types._autogenerated.ome_2016_06.ome.OME


Update the OME state based on proposed changes using JSONPatch, automatically creating missing nodes.


Args
-----=
**```current_state```** :&ensp;<code>OME</code>
:   The current OME state.


**```proposed_change```** :&ensp;<code>List\[Dict\[str, Any]]</code>
:   The change proposed as a JSON Patch document.



Returns
-----=
<code>OME</code>
:   The updated OME state.



Raises
-----=
<code>ValueError</code>
:   If the patch is invalid or cannot be applied, or if the resulting document is not a valid OME model.




    
## Classes


    
### Class `Tee` {#id}




>     class Tee(
>         *streams
>     )


A class to duplicate output to multiple streams.







    
#### Methods


    
##### Method `flush` {#id}




>     def flush(
>         self
>     )




    
##### Method `write` {#id}




>     def write(
>         self,
>         data
>     )





-----
Generated by *pdoc* 0.11.1 (<https://pdoc3.github.io>).
