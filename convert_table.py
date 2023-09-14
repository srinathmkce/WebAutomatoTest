import json
import os
from argparse import ArgumentParser

import pandas as pd
from langchain import PromptTemplate, LLMChain
from langchain.callbacks import wandb_tracing_enabled
from langchain.llms import OpenAI


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
    template = """you will be provided with the information from two input tables and one template table. You need to extract the information from the two input tables and fill the template table. 

    Table 1 - {tableA_row}

    Table 2 - {tableB_row}

    Template - {template_row}

    Construct the template table with following columns - ['Date', 'EmployeeName', 'Plan', 'PolicyNumber', 'Premium'] 

    Follow the instructions carefully:

    Follow the formats specified in the Template
    Provide your response as a JSON object
    If the values are available on both Table 1 and Table 2, choose the first option
    Always transform the data fields to mm-dd-yyyy format

    JSON object:"""

    prompt = PromptTemplate(
        template=template, input_variables=["tableA_row", "tableB_row", "template_row"]
    )

    llm = OpenAI(openai_api_key=OpenAI_API_KEY)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain


def main(args):
    tableA_df, tableB_df, template_df = read_tables(source=args.source, template=args.template)
    llm_chain = create_llm_chain()

    openai_result_list = []
    for i in range(len(tableA_df)):
        tableA_row = tableA_df.iloc[i].to_json()
        tableB_row = tableB_df.iloc[i].to_json()
        template_row = template_df.iloc[i].to_json()
        with wandb_tracing_enabled():
            response = llm_chain.run(
                {"tableA_row": tableA_row, "tableB_row": tableB_row, "template_row": template_row}
            )
        target_row_dict = json.loads(response)
        openai_result_list.append(target_row_dict)

    target_df = pd.DataFrame(openai_result_list)
    target_df.to_csv(args.target, index=False)
    print(target_df)


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
