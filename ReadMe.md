# Exploring AI for Metadata Curation

This GitHub repository contains the code and documentation for the project "Exploring AI for Metadata Curation".
The goal is to design an AI system that can curate OME XML metadata for microscopy images from its raw metadata.

## ToDo

- [x] Fix the evaluation script, to guarantee the global optimal alignment between two XML files
- [ ] Create a report of the evaluation results which compares bioformats and the AI model
- [ ] When running the model, make sure that the output xml is valid
- [x] Think about the idea that TargetTorben looks for fitting ontology terms for the metadata
  - Overkill, lets ignore for now
- [x] Use AICI https://github.com/microsoft/aici to guarantee output formatting
  - Doesnt Work because I dont run the LLM locally
- [ ] Think about using Autogen to have more control over agent communication
- [ ] Register master thesis
- [ ] Use structured prompting to get more control over the output
  - pydantic_test as playground
- [ ] Define realistic goals that I still want to implement

## Problem description

Numerous image-generating techniques, such as microscopy, produce metadata, often in proprietary formats. This presents
challenges in processing (i.e., compatibility with existing pipelines), publishing (i.e., accessibility for readers),
and storage (i.e., future usability of data). Bioformats, a widely used Java library, aims to address these issues by
offering tools to convert these proprietary formats into the open OME format. 
While Bioformats effectively handles many image formats, it encounters difficulties with others, potentially leading to
loss or incorrect translation of metadata. A key issue with Bioformats is its struggle to keep up with the constant
changes in proprietary file formats. Despite these changes often being relatively minor, the effort to maintain
Bioformats is substantial due to the hard-coded and static nature of traditional software paradigms.
Interestingly, format transcription is not dissimilar to language translation, where words in different languages are
mapped to each other. Large language models (LLMs)/ Transformers/ Autoregressive Decoder have recently surprised the scientific community by outperforming
previous state-of-the-art translation tools, despite not being explicitly trained for translation tasks. The vast
amount of multilingual data has enabled these models to develop a deep understanding of language, moving beyond
monolingual word representation towards a generalized, conceptual knowledge base. 
In essence, these models accept words in any language, translate them into an abstract conceptual representation
for reasoning, and then map the resulting concepts back to the desired output language. Applying this logic to
format transcription, the same principle could be used. Instead of directly mapping a property, such as a key-value
pair, from the input format to another property in the output format, the input is first represented as a concept.
This concept can then be transcribed to the specified output. As the conceptual meaning of the input or output does
not derive from literal string representation, this method is resilient to most changes in proprietary file formats.
Therefore, I propose OME Cur-AI-tion, a modified version of LLama2 (Nope). LLama2-based models represent the state-of-the-art
in open-source natural language models (not anymore). Preliminary testing has shown that, through transfer learning, such a model can
standardize output (for example, to the XML syntax) and enhance capabilities in understanding and extracting information
from unstructured plain text proprietary metadata. A fully functional OME Cur-AI-tion has the potential to revolutionize
the landscape of microscopy and unify the efforts of the entire imaging community, much like how the FASTA file format
has enabled the sequence community to work together efficiently.


## Current Standing

- I found that the Mapping of properties is not the fatal flaw of Bioformats, but the lack of target properties in the OME XML schema
- Instead of using LLama2, I am using GPT-4 via api, which has more powerfull function calling capabilities
- Instead of extending Bioformats I could rephrase the task to "substitute Bioformats" and use the AI model to generate the OME XML metadata from scratch
  - This would allow the to use the Bioformats output as "ground truth" for the AI model
  - It could still be formulated that this is done soley for research purposes, and longterm the goal is to extend Bioformats functionality.
  - This experiment could figure out if lLM are able to adhere to a schema.
- 

### Post Presentation 

We discussed two ideas for AI metadata curation:
1. Translate the raw metadata to OME-XML as you would with other languages using a LLM
   1. This can lead to context length issues.
   2. Less automated, since validation and other services still need to be implemented
   
2. Define a system of LLM-Agents, that provide the service of curating metadata
   1. Agent = Autonomously acting LLM (can execute and prompt Agents including itself)
   2. Has long- and short-term memory management to (partially) solve the context length issue
   3.
