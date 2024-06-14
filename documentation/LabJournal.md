
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