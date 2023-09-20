import os
from argparse import ArgumentParser

import pandas as pd
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI


def read_tables(source: str, template: str):
    tableA_path, tableB_path = source.split(",")
    tableA_df = pd.read_csv(tableA_path)
    tableA_df = tableA_df.iloc[-1:]
    tableB_df = pd.read_csv(tableB_path)
    tableB_df = tableB_df.iloc[-1:]
    template_df = pd.read_csv(template)
    return tableA_df, tableB_df, template_df


def create_llm_chain():

    os.environ["WANDB_PROJECT"] = "WebAutomato"
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Set openai key in OPENAI_API_KEY environment variable")
    OpenAI_API_KEY = os.environ["OPENAI_API_KEY"]
    template = """You are an assistant for code generation. Follow the rules strictly given below.

1. I have financial information coming from two different sources (TableA and TableB). The same information might be available in both the tables in different format. For ex: Insurance plan field is available as'Gold Plan' in TableA and the same plan is available as 'Gold Package' in TableB.
2. I have a Template table. The values in the result table which you are going to produce should be similar to the template table.
3. Pick the value from TableA or TableB, but not both. If the value is available in both TableA or TableB, prefer the value from TableA.
4.  Apply the transformation from TableA or TableB to produce the result like template table. For example: The result of the Plan value should be Gold (In TableA it is Gold Plan and TableB it is GoldPackage)
5. Keep the tables open for your reference. TableA and TableB contains millions for rows and hence do not apply merge or join on the TableA and TableB
6.  Write a python pandas code to read all three input tables (table_A.csv, table_B.csv, template.csv)
7.  For each column in the Template table, find how the values from TableA or TableB can be transformed into template column values. Generate the code logic and create a new column in Result table with same name as the template table column.
8. Do not copy the template table. Instead, process column by column and apply necessary transformations.
9. Always transform the date fields to mm-dd-yyyy format
10. Do not rename the column from Source table to achieve the results.
11. Generate only the code and no explanations needed.

TableA:
{tableA_columns}
{tableA_row}

TableB:
{tableB_columns}
{tableB_row}

{template_columns}
{template_row}

Code:

    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "tableA_row",
            "tableA_columns",
            "tableB_row",
            "tableB_columns",
            "template_columns",
            "template_row",
        ],
    )
    # Change the model name based on the context length and accuracy requirements
    llm = ChatOpenAI(openai_api_key=OpenAI_API_KEY, model_name="gpt-3.5-turbo")
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    return llm_chain


def main(args):
    tableA_df, tableB_df, template_df = read_tables(source=args.source, template=args.template)
    llm_chain = create_llm_chain()

    tableA_columns = tableA_df.columns
    tableB_columns = tableB_df.columns
    tableA_row = tableA_df.iloc[0:2].to_json()
    tableB_row = tableB_df.iloc[0:2].to_json()
    template_columns = template_df.columns
    template_row = tableB_df.iloc[:].to_json()

    response = llm_chain.run(
        {
            "tableA_row": tableA_row,
            "tableB_row": tableB_row,
            "template_row": template_row,
            "tableA_columns": tableA_columns,
            "tableB_columns": tableB_columns,
            "template_columns": template_columns,
        }
    )
    print("Response: ", response)

    with open("transform.py", "w") as fp:
        fp.write(response)

    print("Done")


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
