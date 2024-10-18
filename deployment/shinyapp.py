# TODO: Check outputs
# TODO: on click, scroll to the top first highlighted instance
# TODO: Add highlight colours to subject/object
# TODO: add a back button
# TODO: Change to condition or effect
# TODO: Ignore cases with no highlighting

import csv
import re
import sys
from pathlib import Path

from shiny import App, ui, render, reactive

sys.path.append("..")
from useful_string_methods import abbreviate_sci_name
from deployment.base_implementation import read_chunk_file, get_relations_from_taxa_data

chunk_text = read_chunk_file(70)

# Define a list of relationship data
RELATIONSHIPS = get_relations_from_taxa_data('70_gpt-4o_outputs.pickle', 70)


def highlight_phrases(text, sciname, keyword):
    scinames = [sciname, abbreviate_sci_name(sciname)]
    for phrase in scinames:
        pattern = re.compile(f'({re.escape(phrase)})', re.IGNORECASE)
        text = pattern.sub(r'<span class="highlight1" title="Scientific Name">\1</span>', text)

    pattern = re.compile(f'({re.escape(keyword)})', re.IGNORECASE)
    text = pattern.sub(r'<span class="highlight2" title="Condition or Effect">\1</span>', text)
    return text


app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            .highlight1 { background-color: yellow; position: relative; }
            .highlight2 { background-color: lightblue; position: relative; }
            .highlight1:hover::after, .highlight2:hover::after {
                content: attr(title);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                padding: 5px;
                background-color: #333;
                color: white;
                border-radius: 4px;
                white-space: nowrap;
            }
            .container { display: flex; height: 100%; }
            .column-left { flex: 70%; padding: 10px; overflow-y: auto; }
            .column-right { 
                flex: 30%; 
                padding: 10px; 
                position: sticky; 
                top: 0; 
                height: 100vh;
                overflow-y: auto;
            }
        """)
    ),
    ui.h2("Manual Annotation Checker"),
    ui.div(
        {"class": "row"},
        ui.div(
            {"class": "column-left"},
            ui.output_ui("highlighted_text")
        ),
        ui.div(
            {"class": "column-right"},
            ui.h3("Relationship Data"),
            ui.output_text("name1"),
            ui.output_text("relation"),
            ui.output_text("name2"),

            ui.output_text("chunk_id"),

            ui.input_action_button("publish", "Publish"),
            ui.input_action_button("omit", "Pass"),
            ui.output_text("status")
        )
    ),
    ui.tags.script("""
            $(document).on('click', '#omit', function() {
                window.scrollTo({
                  top: 0,
                  left: 0,
                  behavior: "smooth",
                });
            });
        """),
    ui.tags.script("""
            $(document).on('click', '#publish', function() {
                window.scrollTo({
                  top: 0,
                  left: 0,
                  behavior: "smooth",
                });
            });
        """)
)


def server(input, output, session):

    current_relationship = reactive.Value(0)

    @output
    @render.ui
    def highlighted_text():
        highlighted = highlight_phrases(chunk_text, RELATIONSHIPS[current_relationship()][0], RELATIONSHIPS[current_relationship()][1])
        return ui.HTML(f"<p>{highlighted}</p>")

    @output
    @render.text
    def name1():
        return f"Scientific Name: {RELATIONSHIPS[current_relationship()][0]}"

    @output
    @render.text
    def name2():
        return f"Condition or Effect: {RELATIONSHIPS[current_relationship()][1]}"

    @output
    @render.text
    def relation():
        return f"Relation: {RELATIONSHIPS[current_relationship()][2]}"

    @output
    @render.text
    def chunk_id():
        return f"Chunk ID: {RELATIONSHIPS[current_relationship()][3]}"

    @reactive.Effect
    @reactive.event(input.publish)
    def publish_action():
        csv_file = Path("phrases.csv")
        new_row = RELATIONSHIPS[current_relationship()]

        # Append the new row to the CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_row)

    @reactive.Effect
    @reactive.event(input.publish, input.omit)
    def update_relationship():
        current_relationship.set((current_relationship() + 1) % len(RELATIONSHIPS))

    @output
    @render.text
    @reactive.event(input.publish)
    def status():
        return "Published to phrases.csv"

    @output
    @render.text
    @reactive.event(input.omit)
    def status():
        return "Omitted"


app = App(app_ui, server)
