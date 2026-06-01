from ragas.metrics import (
    AnswerAccuracy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
)

from .llm import llm


context_precision = ContextPrecision(llm=llm)
context_recall = ContextRecall(llm=llm)

faithfulness = Faithfulness(llm=llm)
response_relevancy = ResponseRelevancy(llm=llm)

factual_correctness = FactualCorrectness(llm=llm)
answer_accuracy = AnswerAccuracy(llm=llm)