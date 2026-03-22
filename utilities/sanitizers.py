import re
import unicodedata
import html2text

def sanitize_email_text(text):
    """
    Handles different email content sanitizations such as removal of non-printable control characters,
    invisible Unicode characters and different types of os dependent line endings.

    Args:
        text (string): Email content.

    Returns:
        (string): Email contents after removal of above mentioned characters.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)    
    return text

def sanitize_email_html(html_content):
    """
    Handles different sanitizations from emails having html tags.

    Args:
        text (string): Email content.

    Returns:
        (string): Email contents after removal of above mentioned characters.
    """
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = False
    text_maker.ignore_images = True
    text_maker.ignore_emphasis = False
    text_maker.ignore_tables = False
    text_maker.body_width = 0
    clean_text = text_maker.handle(html_content)
    return clean_text

