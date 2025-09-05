import argparse
import os
from pydantic import BaseModel, Field
from datasets import Dataset

from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration


################################################################################
# Script Parameters
################################################################################

parser = argparse.ArgumentParser(
    description="Generate exam questions from text files in a directory."
)
parser.add_argument(
    "--model_id",
    type=str,
    default="Qwen/Qwen2.5-7B-Instruct",
    help="Model ID for text generation",
)
parser.add_argument(
    "--tokenizer_id",
    type=str,
    default="Qwen/Qwen2.5-7B-Instruct",
    help="Tokenizer ID for text generation",
)
parser.add_argument(
    "--input_dir",
    type=str,
    help="Directory containing input text files",
    default="data",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=2048,
    help="Maximum number of new tokens to generate",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="exam_questions_output",
    help="Directory to save the generated datasets",
)

args = parser.parse_args()

################################################################################
# Load the documents
# We assume that the documents are in the input directory, and that each file
# is a separate document about the same topic.
################################################################################

# Process all text files in the input directory
documents = []
for filename in os.listdir(args.input_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(args.input_dir, filename)
        with open(file=file_path, mode="r", encoding="utf-8") as file:
            document_content = file.read()
            documents.append(document_content)

# Create a single dataset from all document contents
dataset = Dataset.from_dict({"document": documents})

################################################################################
# Define the prompts
# We use a system prompt to guide the model to generate the correct output format.
# A template is used to insert the document into the prompt.
################################################################################

SYSTEM_PROMPT = """\
You are an exam writer specialized in writing exams for students.
Your goal is to create questions and answers based on the document provided, 
and a list of distractors, that are incorrect but viable answers to the question.
Your answer must adhere to the following format:
```
[
    {
        "question": "Your question",
        "answer": "The correct answer to the question",
        "distractors": ["wrong answer 1", "wrong answer 2", "wrong answer 3"]
    },
    ... (more questions and answers as required)
]
```
""".strip()

INSTRUCTION_TEMPLATE = """\
    Generate a list of answers and questions about the document. 
    Document:\n\n{{ instruction }}"""

################################################################################
# Define the output structure
# We define a data model for the output of the pipeline, this is used to ensure
# that the output is in the correct format for the evaluation task.
################################################################################


class ExamQuestion(BaseModel):
    question: str = Field(..., description="The question to be answered")
    answer: str = Field(..., description="The correct answer to the question")
    distractors: List[str] = Field(
        ..., description="A list of incorrect but viable answers to the question"
    )


class ExamQuestions(BaseModel):
    exam: List[ExamQuestion]


################################################################################
# Create the pipeline
# We create a pipeline with a single task that generates the exam questions
# based on the document and in the correct format. We will Hugging Face
# InferenceEndpoints and the model specified in the arguments.
################################################################################

with Pipeline(
    name="Domain-Eval-Questions",
    description="Generate exam questions based on given documents.",
) as pipeline:
    # Set up the text generation task
    text_generation = TextGeneration(
        name="exam_generation",
        llm=InferenceEndpointsLLM(
            model_id=args.model_id,
            tokenizer_id=args.model_id,
            api_key=os.environ["HF_TOKEN"],
            structured_output={
                "schema": ExamQuestions.model_json_schema(),
                "format": "json",
            },
        ),
        input_batch_size=8,
        output_mappings={"model_name": "generation_model"},
        input_mappings={"instruction": "document"},
        system_prompt=SYSTEM_PROMPT,
        template=INSTRUCTION_TEMPLATE,
    )


################################################################################
# Run the pipeline
# We run the pipeline for all documents and save the results to the output path.
################################################################################

if __name__ == "__main__":
    # Run the pipeline for all documents
    distiset = pipeline.run(
        parameters={
            "exam_generation": {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": args.max_new_tokens,
                    }
                }
            }
        },
        use_cache=False,
        dataset=dataset,
    )

    distiset.save_to_disk(args.output_path)
