import openai

openai.api_key = "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"


data = "Many image-generating methods, such as microscopy, output the metadata in a proprietary format. This causes problems in processing (can my pipeline use that image), publishing (can the reader access the data), and storage (will the data be usable in the future). Bioformats, a widely distributed java library, attempts to tackle those problems by providing tools to translate to the open OME format. Currently, Bioformats works well, with many image formats, but struggles with some. This can lead to metadata getting lost, or falsely translated. Fundamental to the problems of bioformats is the moving target. Provider of proprietary file formats constantly change said formats, making the effort to maintain bioformats enormous, despite changes often being relatively minor. This is caused by the hard coded and static nature of the classical software pardigm. In the most general view format transkription is not much different to language translation, in that words in both languages  are mapped to one another. Translation is a task at which large language models (LLM) surprised the wider scientific community, in that they outperformed previous state of the art translation tools, without being explicitly trained for translation tasks. The sheer amount of multilingual data allowed these models to have a deep understanding of language, going beyond monolingual representation of words, towards a generalised and concept based knowledge base. In other words the models take words in any language as input, translate these to an abstract conceptional representation in which they reason and then map the resulting concepts back to the desired output language. Applying this to the format transcription problem, I believe, that the same logic applies. The input format provides a property such as a key value pair, instead of mapping this directly to another property in the output format, the input is represented as concept first. The concept can then be transcribed to the specified output. Since the conceptual meaning of the input or output doesnt stem from literal string representation, this method is resilient to most changes in proprietary file formats. Thats why here I propose OME Cur-AI-tion a modified version of LLama2. LLama2 based models represent the state of the art open source natural language models. Initial testing showed, that via transfer learning  such a model is cabable to: standardize output (for example to the XML syntax) and increase capabilities in understanding and extracting information from unstructured plain text proprietary metadata. A working OME Cur-AI-tion has the potential to sustainably change the landscape of microscopy and reunite the efforts of the entire imaging community, similar to how the FASTA file format allows the sequence community to work together efficiently."

messages=[
    {"role": "system", "content": "You are a helpful assistant and proffesional scientific writer. You excel at proof reading and improving scientific manuscripts. You are also a skilled software developer and have a deep understanding of the OME file format."},
    {"role": "user", "content": f"Please help improve and correct the following text: \n {data}"},
]

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
    temperature=0,
    max_tokens=5000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["\"\"\""]
)
print(response["choices"][0]["message"]["content"])