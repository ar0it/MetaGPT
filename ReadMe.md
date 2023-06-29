# Exploring AI for Metadata Curation

## Introduction

In the presentation (13.06.23) introducing the XML-Editor, we discussed possibilities to automate the
process of curating metadata using technologies like large language models (LLM) and more specifically,
tools like AutoGPT [1]. We came to the conclusion, that testing is needed to get an idea of how
to approach the topic best. In this short report, I want to document the current progress, which
exploratory experiments have been attempted, and lay out a roadway on how to proceed. The report
is structured into three parts. Starting with the ”Problem description” part, where I want to formulate
the problem and give background information. Next, I want to present the up-to-date, takeaway
message in the ”Current Standing” part. Lastly, in the ”Experiment-Diary” part, I want to collect
methods and results, in a diary-like fashion.

## Problem description

Many image-generating methods, such as microscopy, output the metadata in a proprietary format.
This causes problems in processing (can my pipeline use that image), publishing (can the reader access
the data), and storage (will the data be usable in the future). Bioformats [2] attempts to tackle those
problems, by providing tools to translate to the open OME [3] format. Currently, Bioformats works
well, with many images, but struggles with some formats. This causes metadata to get lost, or falsely
translated. I call the in-place correction (not via annotation) of that erroneous metadata ”curation”.

## Current Standing

### Post Presentation 

We discussed two ideas for AI metadata curation:
1. Translate the raw metadata to OME-XML as you would with other languages using a LLM
   1. This can lead to context length issues.
   2. Less automated, since validation and other services still need to be implemented
   
2. Define a system of LLM-Agents, that provide the service of curating metadata
   1. Agent = Autonomously acting LLM (can execute and prompt Agents including itself)
   2. Has long- and short-term memory management to (partially) solve the context length issue
   3.
