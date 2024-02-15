from bs4 import BeautifulSoup
import pandas as pd

with open("index_shelly.html") as fp:
    html_doc = fp.read()

soup = BeautifulSoup(html_doc, "html.parser")

docs = soup.find_all("div", class_="theme-doc-markdown markdown")[0]

functions_df = pd.DataFrame(
    columns=[
        "function_name",
        "description",
        "request_properties",
        "request_notes",
        "response_properties",
        "response_notes",
    ]
)

component_df = pd.DataFrame(columns=["name", "type", "description"])

import sys

redundant_notes = [
    "Attributes in the result:",
    "Attributes in the result (only the ones available are shown):",
    "Parameters:",
]


def parse_table(table):
    properties_list = []
    for row in table.css.select("table > tbody > tr"):
        col_values = []
        for child in row.css.select("tr > td"):
            val = child.find("p").get_text()
            col_values.append(val)
        properties_list.append(
            {"name": col_values[0], "type": col_values[1], "description": col_values[2]}
        )
    return properties_list


def parse_list(tag):
    str_items = []
    for li in tag.css.select("ul > li:not(li li)"):
        item = ""
        subitems = []
        for ch in li.children:
            if ch.name != "ul":
                item += ch.get_text()
            else:
                subitems.append(ch.get_text())
        item += " " + ".".join(subitems) + "."
        str_items.append(item)
    str_items = " ".join(str_items)
    return str_items


functions_started = False
frame_name = None
function_dict = {
    "function_name": "",
    "description": "",
    "request_properties": None,
    "request_notes": "",
    "response_properties": None,
    "response_notes": "",
}
service_dict = {
    "name": "",
    "type": "",
    "description": "",
}

for child in docs.children:
    print(child)
    break
    if child.name == "h1":
        service_dict["name"] = child.get_text()
    elif child.name == "p" and not functions_started:
        service_dict["description"] = child.get_text()
        if "service" in service_dict["description"]:
            service_dict["type"] = "service"
        else:
            service_dict["type"] = "component"
        component_df.loc[len(component_df)] = service_dict

    if child.name != "h2" and not functions_started:
        continue
    elif child.name == "h2":
        if function_dict["function_name"] != "":
            functions_df.loc[len(functions_df)] = function_dict
            function_dict = {
                "function_name": "",
                "description": "",
                "request_properties": None,
                "request_notes": "",
                "response_properties": None,
                "response_notes": "",
            }
            frame_name = None

        function_name = str(child.contents[0])
        if "HTTP Endpoint" in function_name:
            break

        function_dict["function_name"] = str(child.contents[0])
        functions_started = True
    elif child.name == "p" and frame_name is None:
        function_dict[f"description"] = child.get_text()
    elif child.name == "h4":
        frame_name = str(child.contents[0]).lower()
    elif child.name == "p":
        if child.contents[0] in redundant_notes:
            continue
        function_dict[f"{frame_name}_notes"] += child.get_text()
    elif child.name == "ul":
        function_dict[f"{frame_name}_notes"] += parse_list(child)
    elif child.name == "table":
        function_dict[f"{frame_name}_properties"] = parse_table(child)
    else:
        with open("other_tags.html", "a") as f:
            f.write(child.prettify())

functions_df.to_csv("functions.csv", index=False)
component_df.to_csv("component.csv", index=False)
