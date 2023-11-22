import pandas as pd
import config


# acronyms cleaning functions
def convert_list_to_categorical(acronym_list: list[str]) -> tuple[str, str, str]:
    """
    Converts a list of acronyms into a tuple of strings,
    the first 2 strings in the list get mapped to the first 2 values in the tuple and the remaining strings in the list are joined with a spaced and mapped to the 3rd value in the tuple.
    :param acronym_list: list of acronyms
    :return: tuple of strings
    """
    if len(acronym_list) == 1:
        return acronym_list[0], "", ""
    elif len(acronym_list) == 2:
        return acronym_list[0], acronym_list[1], ""
    else:
        return acronym_list[0], acronym_list[1], ' '.join(acronym_list[2:])


def preprocess_node_id(node: str, mappings: dict) -> str:
    """
    :param node: node string to preprocess
    :param mappings: dictionary of values to replace
    :return: preprocessed node string
    """
    # convert to string
    output = str(node)

    # replace from mappings
    for key, value in mappings.items():
        output = output.replace(key, value)

    # if the next letter is capitalised, add a space
    for i in range(len(output) - 1):
        if output[i].islower() and output[i + 1].isupper():
            output = output[:i + 1] + " " + output[i + 1:]

    # if the word "Temp" appears, add a space before it
    if "Temp" in output:
        output = output.replace("Temp", " Temp")

    # output should be only letters and spaces
    output = ''.join([i for i in output if i.isalpha() or i == " "])

    # convert any double spaces to single spaces
    output = output.replace("  ", " ")

    # lowercase the string
    output = output.lower()

    return output


def replace_to_full_words(text: str, mappings: dict) -> str:
    output = str(text)

    # lowercase the string just in case
    output = output.lower()

    for key, value in mappings.items():
        output = output.replace(key, value)

    return output


def preprocess_unit(unit: str) -> str:
    """
    :param unit: unit string to preprocess
    :return: preprocessed unit string
    """
    # convert to string
    output = str(unit)
    # lowercase the string
    output = output.lower()

    # replace 0 with nicht bekannt
    output = output.replace("0", "nicht bekannt")

    # replace nan with nicht bekannt
    output = output.replace("nan", "nicht bekannt")

    # if mwh, convert to kwh
    output = output.replace("mwh", "kwh")

    # replace     # replace nan with nicht bekannt
    output = output.replace("nan", "nicht bekannt")

    output = output.replace("m3", "m³")

    return output


def add_texts(node_id: str, name_cust: str, unit: str) -> str:
    """
    Adds the node_id, name_cust and unit strings together
    :param node_id: node_id string
    :param name_cust: name_cust string
    :param unit: unit string
    :return: concatenated string
    """
    output = f"knoten: {node_id}, name: {name_cust}, einheit: {unit}"
    return output


def main():
    df = pd.read_excel(config.ORIGINAL_DATA_PATH)

    # clean acronyms
    df = df.dropna(subset=["acronym_n"])
    df = df[df['acronym_n'] != 0]

    # remove the str '(n)' from the acronym column
    df['acronym_n'] = df['acronym_n'].str.replace(r"(n)", "")

    # split the acronym column into a list of acronyms
    df['acronym_split'] = df['acronym_n'].str.split('_')

    # convert the list of acronyms into a tuple of strings
    measurement, location, description = zip(*df['acronym_split'].apply(convert_list_to_categorical))
    df['measurement'] = measurement
    df['location'] = location
    df['description'] = description

    # preprocess node_id
    df['node_text'] = df['node_id'].apply(preprocess_node_id, mappings=config.NODE_ID_MAPPINGS)

    # preprocess name_cust
    df['name_cust_text'] = df['name_cust'].apply(preprocess_node_id, mappings=config.NAME_CUST_DICT)

    # replace node and name with full words
    df['node_text'] = df['node_text'].apply(replace_to_full_words, args=(config.GENERAL_MAPPINGS,))
    df['name_cust_text'] = df['name_cust_text'].apply(replace_to_full_words, args=(config.GENERAL_MAPPINGS,))

    # preprocess unit
    df['unit_text'] = df['unit_getec'].apply(preprocess_unit)

    # unify texts
    df['text'] = df.apply(lambda x: add_texts(x['node_text'], x['name_cust_text'], x['unit_text']), axis=1)

    # drop unnecessary columns
    data = df[['text', 'measurement', 'location', 'description']].dropna()

    # convert measurement, location and description to categorical
    data['measurement'] = data['measurement'].astype('category')
    data['location'] = data['location'].astype('category')
    data['description'] = data['description'].astype('category')

    # save the cleaned data
    data.to_csv(config.CLEAN_DATA_PATH, index=False)
    data.to_pickle("data/sensors_cleaned.pkl")

    print("Data saved to: ", config.CLEAN_DATA_PATH)


if __name__ == '__main__':
    main()


