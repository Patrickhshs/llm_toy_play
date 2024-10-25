# llm_toy_play

# To-dos:

- Instruction tuning
- Try to solve a math problem
- collect the conversation dataset between two LMs
- Speaker 1 (knowing the answer in advance) helps speaker 2 to find the answer.
- goal for each speaker:
    - Speaker 1 shouldn't reveal the answer in the process until speaker 2 speak out the answer.
    - Speaker 2 aims at following the question/guidance from the speaker 1 correctly.
    - formulating as a cooperative game?
    - how to design the loss function to avoid being lazy to question and don't lose the ability so solve the question by itself.
    - come up with the metric on how to evaluate the dialogue of teacher LM 1/
