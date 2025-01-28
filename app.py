import gradio as gr
import spacy
import re
from functools import lru_cache
import logging
import os

# -------------------- Setup Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    filename='masking.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------- Entity Type Mapping --------------------
# Mapping from friendly names to spaCy labels
friendly_to_label = {
    "Person": "PERSON",
    "Organization": "ORG",
    "Geopolitical Entity": "GPE",
    "Location": "LOC",
    "Date": "DATE",
    "Money": "MONEY",
    "Event": "EVENT",
    "Law": "LAW",
    "Language": "LANGUAGE",
    "Cardinal": "CARDINAL",
    "Ordinal": "ORDINAL",
    "Facility": "FAC",
    "Product": "PRODUCT",
    "NORP": "NORP",  # Nationalities or religious or political groups
}

# List of available entity types for the UI
available_entity_types = [
    ("Person", "PERSON"),
    ("Organization", "ORG"),
    ("Geopolitical Entity", "GPE"),
    ("Location", "LOC"),
    ("Date", "DATE"),
    ("Money", "MONEY"),
    ("Event", "EVENT"),
    ("Law", "LAW"),
    ("Language", "LANGUAGE"),
    ("Cardinal", "CARDINAL"),
    ("Ordinal", "ORDINAL"),
    ("Facility", "FAC"),
    ("Product", "PRODUCT"),
    ("NORP", "NORP"),
]

# Extract friendly names for the CheckboxGroup
entity_type_choices = [name for (name, label) in available_entity_types]

# -------------------- SpaCy Model Caching --------------------
@lru_cache(maxsize=3)
def load_spacy_model(model_name):
    """
    Load and cache the specified spaCy model.
    
    Args:
        model_name (str): Name of the spaCy model to load.
    
    Returns:
        spacy.Language: Loaded spaCy language model or None if loading fails.
    """
    try:
        return spacy.load(model_name)
    except Exception as e:
        logging.error(f"Error loading model '{model_name}': {e}")
        return None

# -------------------- Masking Function --------------------
def mask_entities_and_contacts(model_name, text, file_path, dynamic_mask, entity_type_selection):
    """
    Masks selected entities and sensitive information in the input text.
    
    Args:
        model_name (str): Name of the spaCy model to use.
        text (str): Text input from the user.
        file_path (str): Path to the text file input.
        dynamic_mask (bool): Flag to determine masking strategy.
        entity_type_selection (list): List of selected entity types to mask.
    
    Returns:
        tuple: Masked text and feedback message.
    """
    # -------------------- Handle File Path Input --------------------
    if file_path:
        if not os.path.isfile(file_path):
            logging.error(f"File path '{file_path}' does not exist.")
            return f"Error: The file path '{file_path}' does not exist.", ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logging.error(f"Error reading file at '{file_path}': {e}")
            return f"Error reading the file at '{file_path}': {e}", ""
    
    # -------------------- Validate Input --------------------
    if not text:
        return "Please provide input text or a valid file path.", ""
    
    # -------------------- Load SpaCy Model --------------------
    nlp = load_spacy_model(model_name)
    if not nlp:
        return f"Failed to load the spaCy model: {model_name}", ""
    
    # -------------------- Process Text with SpaCy --------------------
    doc = nlp(text)
    
    # -------------------- Map Friendly Names to Labels --------------------
    entity_labels = [friendly_to_label.get(name) for name in entity_type_selection]
    # Remove any None values in case of unrecognized names
    entity_labels = [label for label in entity_labels if label]
    
    # -------------------- Mask Entities --------------------
    masked_text = text
    masked_entities = []
    
    # Sort entities in reverse order to prevent index shifting
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        if ent.label_ in entity_labels:
            start, end = ent.start_char, ent.end_char
            masked_entity = '*' * (end - start) if dynamic_mask else "*****"
            masked_text = masked_text[:start] + masked_entity + masked_text[end:]
            masked_entities.append((ent.text, ent.label_))
            logging.info(f"Masked entity: '{ent.text}' ({ent.label_}) at positions {start}-{end}")
    
    # -------------------- Mask Sensitive Information via Regex --------------------
    # Define regex patterns
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    ssn_regex = r'\b\d{3}-\d{2}-\d{4}\b'
    phone_regex = r'\b\d{3}-\d{3}-\d{4}\b'
    credit_card_regex = r'\b(?:\d[ -]*?){13,16}\b'  # Simple regex for credit card numbers
    
    # Mask email addresses
    masked_text, num_emails = re.subn(email_regex, '*****', masked_text)
    if num_emails > 0:
        logging.info(f"Masked {num_emails} email address(es).")
    
    # Mask SSNs
    masked_text, num_ssns = re.subn(ssn_regex, '*****', masked_text)
    if num_ssns > 0:
        logging.info(f"Masked {num_ssns} SSN(s).")
    
    # Mask phone numbers
    masked_text, num_phones = re.subn(phone_regex, '*****', masked_text)
    if num_phones > 0:
        logging.info(f"Masked {num_phones} phone number(s).")
    
    # Mask credit card numbers
    masked_text, num_cards = re.subn(credit_card_regex, '*****', masked_text)
    if num_cards > 0:
        logging.info(f"Masked {num_cards} credit card number(s).")
    
    # -------------------- Prepare Feedback Message --------------------
    feedback_messages = []
    
    # Feedback for masked entities
    if masked_entities:
        entity_details = ", ".join([f"'{ent[0]}' ({ent[1]})" for ent in masked_entities])
        feedback_messages.append(f"Masked {len(masked_entities)} entity(ies): {entity_details}.")
    else:
        feedback_messages.append("No entities were masked based on your selection.")
    
    # Feedback for regex-based masking
    if num_emails > 0:
        feedback_messages.append(f"Masked {num_emails} email address(es).")
    if num_ssns > 0:
        feedback_messages.append(f"Masked {num_ssns} SSN(s).")
    if num_phones > 0:
        feedback_messages.append(f"Masked {num_phones} phone number(s).")
    if num_cards > 0:
        feedback_messages.append(f"Masked {num_cards} credit card number(s).")
    
    # General feedback if nothing was masked
    if not any([masked_entities, num_emails, num_ssns, num_phones, num_cards]):
        feedback_messages.append("No sensitive information was found to mask.")
    
    feedback = "\n".join(feedback_messages)
    
    return masked_text, feedback

# -------------------- Build Gradio UI --------------------
with gr.Blocks() as app:
    # -------------------- Header: Title and Description --------------------
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<h1 style='text-align: center;'>🛡️ Classy Scrubber</h1>")
            gr.Markdown("<h3 style='text-align: center;'>Gettin NERd with spaCy</h3>")
    
    gr.HTML("<hr/>")  # Horizontal separator
    
    # -------------------- First Row: Input and Output --------------------
    with gr.Row(equal_height=True):
        # Input Text
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                lines=15,
                placeholder="Enter Text Here...",
                label="📄 Input Text",
                interactive=True
            )
        
        # Scrubbed Output
        with gr.Column(scale=1):
            output = gr.Textbox(
                lines=15,
                placeholder="Scrubbed Output...",
                label="📝 Scrubbed Output",
                interactive=False
            )
    
    # -------------------- Second Row: Controls --------------------
    with gr.Row():
        # Left Column: File Path Input, Model Selection, Dynamic Masking (1/3)
        with gr.Column(scale=1):
            file_path_input = gr.Textbox(
                placeholder="Enter file path here...",
                label="📂 File Path",
                interactive=True,
                elem_id="file-path-input"
            )
            model_choice = gr.Dropdown(
                choices=["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
                value="en_core_web_sm",
                label="🧠 Select a spaCy Model",
                interactive=True
            )
            dynamic_mask = gr.Checkbox(
                label="Use dynamic masking (mask with '*' matching entity length)",
                value=True
            )
        
        # Right Column: Select Entity Types to Redact (2/3)
        with gr.Column(scale=2):
            entity_type_selection = gr.CheckboxGroup(
                choices=entity_type_choices,
                value=["Person", "Organization", "Geopolitical Entity"],
                label="🗂️ Select entity types to redact"
            )
    
    # -------------------- Third Row: Scrub Button --------------------
    with gr.Row():
        scrub_button = gr.Button("🧹 Scrub", variant="primary")
    
    # -------------------- Fourth Row: Feedback --------------------
    with gr.Row():
        feedback = gr.Textbox(
            label="Feedback",
            lines=4,
            interactive=False
        )
    
    # -------------------- Wire up Inputs and Outputs --------------------
    scrub_button.click(
        fn=mask_entities_and_contacts,
        inputs=[
            model_choice,
            text_input,
            file_path_input,
            dynamic_mask,
            entity_type_selection
        ],
        outputs=[output, feedback]
    )

# Launch the Gradio app
app.launch()
