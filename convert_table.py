import json
import os
from argparse import ArgumentParser

import pandas as pd
from langchain import PromptTemplate, LLMChain
from langchain.callbacks import wandb_tracing_enabled
from langchain.chat_models import ChatOpenAI


def read_tables(source: str, template: str):
    tableA_path, tableB_path = source.split(",")
    tableA_df = pd.read_csv(tableA_path)
    tableB_df = pd.read_csv(tableB_path)
    template_df = pd.read_csv(template)
    return tableA_df, tableB_df, template_df


def create_llm_chain():

    os.environ["WANDB_PROJECT"] = "WebAutomato"
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Set openai key in OPENAI_API_KEY environment variable")
    OpenAI_API_KEY = os.environ["OPENAI_API_KEY"]
    template = """you are an assistant. you will be provided with the information from two input tables and one template table. You need to extract the information from the two input tables and fill the result table.  

    Table 1 - {tableA_row}

    Table 2 - {tableB_row}

    Template - {template_row}

    Follow the instructions carefully:
    The result row contains exactly the same number of rows as Table 1 and do not fill any additional data
    Construct the result table with the same columns and format as template table.
    Always transform the date fields to mm-dd-yyyy format
    Follow the formats specified in the Template
    Provide your response as a JSON object
    If the values are available on both Table 1 and Table 2, choose the first option
    
    JSON object:"""

    prompt = PromptTemplate(
        template=template, input_variables=["tableA_row", "tableB_row", "template_row"]
    )
    # Change the model name based on the context length and accuracy requirements
    llm = ChatOpenAI(openai_api_key=OpenAI_API_KEY, model_name="gpt-3.5-turbo")
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    return llm_chain


def main(args):
    tableA_df, tableB_df, template_df = read_tables(source=args.source, template=args.template)
    llm_chain = create_llm_chain()

    openai_result_list = []
    error_records = []
    # Set the batch size based on the table data
    # In case the columns text are shorter, increase the batch size
    # If the column contains large string, reduce the batch size accordingly
    batch_size = 5

    for batch_start in range(0, len(tableA_df), batch_size):
        batch_end = min(batch_start + batch_size, len(tableA_df))
        tableA_row = tableA_df.iloc[batch_start:batch_end].to_json()
        tableB_row = tableB_df.iloc[batch_start:batch_end].to_json()
        # As the template contains only few rows, take all of them
        # if there are too many rows, just fetch first n samples - template_df.iloc[:10].to_json()
        template_row = template_df.iloc[:].to_json()
        try:
            with wandb_tracing_enabled():
                response = llm_chain.run(
                    {
                        "tableA_row": tableA_row,
                        "tableB_row": tableB_row,
                        "template_row": template_row,
                    }
                )
            target_row_dict = json.loads(response)
            target_sub_df = pd.DataFrame(target_row_dict)
            openai_result_list.append(target_sub_df)
        except Exception as e:
            # Capture the Table A record that caused the exception
            error_records.append(tableA_row)

    target_df = pd.concat(openai_result_list, ignore_index=True)
    target_df.to_csv(args.target, index=False)
    # Create an email to notify error records
    print("Error Records: ", error_records)
    print("Final Output: ", target_df)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        type=str,
        help="Path to Table A and Table B",
    )

    parser.add_argument(
        "--template",
        required=True,
        type=str,
        help="Path to Template table",
    )

    parser.add_argument(
        "--target",
        default="output.csv",
        type=str,
        help="Path to write target table",
    )

    main(parser.parse_args())
