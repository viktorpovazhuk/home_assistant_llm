from bs4 import BeautifulSoup
import pandas as pd
import json

# def parse_table(table):
#     properties_list = []
#     for row in table.css.select("table > tbody > tr"):
#         col_values = []
#         for child in row.css.select("tr > td"):
#             val = child.find("p").get_text()
#             col_values.append(val)
#         properties_list.append(
#             {"name": col_values[0], "type": col_values[1], "description": col_values[2]}
#         )
#     return str(properties_list)


def parse_error_table(table):
    errors = []
    for row in table.tbody.contents:
        values = []
        for col in row.contents:
            el = col.find("p")
            if el is None:
                val = "boolean"
            else:
                val = el.get_text()
            values.append(val)
        name = values[0]
        condition = values[1]
        errors.append(f"Error: {name}. Condition: {condition}")
    errors = ".\n".join(errors)
    return errors


def parse_property_table(table):
    properties = {}
    for prop_row in table.tbody.contents:
        prop_values = []
        for prop_col in prop_row.contents:
            el = prop_col.find("p")
            if el is None:
                val = "boolean"
            else:
                val = el.get_text()
            prop_values.append(val)
        property_name = prop_values[0]
        prop_dict = {"type": prop_values[1], "description": prop_values[2]}
        if prop_dict["type"] == "object":
            embeded_table = prop_row.contents[2].find("table")
            if embeded_table is not None:
                prop_dict["properties"] = parse_table(embeded_table)
        properties[property_name] = prop_dict
    return properties


def parse_table(table):
    table_type = table.thead.tr.th.get_text()
    if table_type == "Error":
        return parse_error_table(table)
    elif table_type == "Property":
        return json.dumps(parse_property_table(table))
    else:
        print("Unknown table type")
        return ""


def parse_list(list_el):
    items = []
    for li in list_el.contents:
        item = ""
        for ch in li.children:
            if ch.name != "ul":
                item += ch.get_text() + " "
            else:
                item += "\n" + parse_list(ch) + "\n"
        item = item.strip()
        items.append(item)
    items = ".\n".join(items)
    return items


def parse_note(div_el):
    heading_el = div_el.contents[0]
    content_el = div_el.contents[1]
    content_text = ""
    for child in content_el.contents:
        if child.name == "ul":
            content_text += parse_list(child)
        else:
            content_text += child.get_text()
    text = heading_el.get_text().capitalize() + ". " + content_text
    return text


def parse_code(div_el):
    return div_el.get_text(strip=True)


# inplace
def init_function_dict(di):
    empty_dict = {"name": "", "description": ""}
    for k, v in empty_dict.items():
        di[k] = v
