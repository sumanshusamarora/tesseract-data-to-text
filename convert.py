import pandas as pd
import numpy as np

def generate_text_from_ocr_output(
    ocr_output_path: str, text_join_delimiter="\n", overlap=0.3
):
    """
    Reads OCR json output and generates ocr text from it
    """
    ocr_dataframe = pd.read_json(ocr_output_path)
    ocr_dataframe = ocr_dataframe[ocr_dataframe["height"] < ocr_dataframe["top"].max()]
    ocr_dataframe["bottom"] = ocr_dataframe["top"] + ocr_dataframe["height"]
    ignore_index = []
    line_indexes = []
    data_indexes = list(ocr_dataframe.index)
    for i in data_indexes:
        if (
            i not in ignore_index
            and ocr_dataframe["text"][i]
            and ocr_dataframe["text"][i].strip() != ""
        ):
            this_row_bottom = ocr_dataframe["top"][i] + ocr_dataframe["height"][i]
            line_index = list(
                ocr_dataframe[
                    (~ocr_dataframe.index.isin(ignore_index))
                    & (
                        ocr_dataframe["top"]
                        <= this_row_bottom - (overlap * ocr_dataframe["height"][i])
                    )
                    & (
                        ocr_dataframe["bottom"]
                        >= ocr_dataframe["top"][i]
                        + (overlap * ocr_dataframe["height"][i])
                    )
                ]
                .sort_values(by="left")
                .index
            )
            ignore_index += line_index
            line_indexes.append(line_index)

    all_tops = [ocr_dataframe.iloc[index_l[0]]["top"] for index_l in line_indexes]
    line_indexes = [line_indexes[ind] for ind in np.argsort(all_tops)]

    text_list = [
        " ".join(
            [
                ocr_dataframe["text"][index]
                for index in line_index
                if ocr_dataframe["text"][index]
            ]
        )
        for line_index in line_indexes
    ]

    return text_join_delimiter.join(text_list)


def stitch_ocr_files(filepaths: list, text_join_delimiter="\n"):
    """
    Open filepaths and read text. Finally return concatenated text
    """
    document_text_list = []
    for filepath in filepaths:
        ocr_text = generate_text_from_ocr_output(
            ocr_output_path=filepath, text_join_delimiter=text_join_delimiter,
        )
        document_text_list.append(ocr_text)

    document_ocr_text = text_join_delimiter.join(document_text_list)
    return document_ocr_text
