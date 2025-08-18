# mdx_generator.py
import datetime
import openai
from text_find import find_homily
import json
import os

# Set your OpenAI API key
api_key = "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"
openai.api_key = api_key

def mdx_generator(homily_text):
    """
    Generate an MDX page from the provided homily transcription text.
    
    This is the text of a very long speech, usually more than 10 minutes long. The transcription text must remain 100% unchanged.
    Do not clip any of the text. The generated MDX page will:
      - Include a dynamic front matter section with generated fields (PageTitle, MetaDescription, etc.), ensuring that all values are wrapped in quotes.
      - Start with a "Summary of Headings" (table of contents with anchor links).
      - Divide the transcription into logical sections with meaningful headings/subheadings.
      - End with a 2–3 paragraph summary recapping the content.
      - Not mention the priest's name unless it is part of the text.
      - Include a blockquote with the offertory at the top.
      - Optionally include short Douay-Rheims Epistle and Gospel quotes (if relevant), based on the Sunday prior to today's date.
      - Focus on SEO, ensuring the Title is strong and the keywords include not only topics of the homily but also terms like "The Latin Mass", "Tridentine Mass", and "Traditional Catholic".
      - Include YouTube-specific fields: 
          - youtube_description: A detailed, engaging YouTube description in 3–4 paragraphs that begins with the following call-to-action text exactly as shown:
            
            Please click on the link to Contribute to our project.
            https://www.mylatinmass.com/donate

            Thank you. All contributions are greatly appreciated.
            - - -
            ABOUT THIS VIDEO:
            
            Followed by a rich description of the lecture.
          - youtube_hash: A comma-separated list of hashtags relevant to the lecture.
    
    Parameters:
      homily_text (str): The complete, unedited transcription text.
    
    Returns:
      str: The MDX page content.
    """
    # Calculate dynamic dates for the front matter.
    today = datetime.date.today()
    # Calculate the previous Sunday (if today is Sunday, it will return today).
    offset = (today.weekday() + 1) % 7
    previous_sunday = today - datetime.timedelta(days=offset)
    date_str = previous_sunday.strftime("%Y-%m-%d")  # previous Sunday
    mod_date_str = today.strftime("%Y-%m-%d")         # today

    # Build the prompt for GPT-4.
    prompt = f"""
You are an expert MDX formatter. Your task is to transform the provided transcription text (which must remain 100% unchanged)
into a complete MDX page with logical headings and subheadings. In addition, generate a dynamic front matter section based on an analysis of the text.
Do not alter or remove any words from the transcription. Keep in mind that this is a Catholic homily that may include religious phrases, Ecclesiastical Latin, and references relevant to Traditional Catholics who follow the Tridentine Mass.

Requirements:
1. At the very top, insert a "Summary of Headings" table of contents with anchor links to each section.
2. Insert meaningful section headings and subheadings throughout the text to break it into logical, readable sections.
3. Include a final summary in 2–3 paragraphs that concisely recaps the text.
4. Generate dynamic front matter fields by analyzing the text. Each front matter value MUST be wrapped in quotes. For example, use:
---
title: "{{PageTitle}}"
description: "{{MetaDescription}}"
keywords: "{{YoutubeTags}}"
youtube_description: "{{YoutubeDescription}}"
youtube_hash: "{{YoutubeHash}}"
mdx_file: "src/mds/{{PageTitleInKebabCase}}.mdx"
category: "lectures"
slug: "/{{PageTitleInKebabCase}}"
date: "{date_str}" 
modDate: "{mod_date_str}"
author: "Fr. Gerrity"
media_type: "video"
media_path: "{{YoutubeVideoId}}"
media_title: "{{PageTitle}}"
media_alt: "{{MediaAltText}}"
media_aria: "{{MediaAriaLabel}}"
prev_topic_label: ""
prev_topic_path: ""
next_topic_label: ""
next_topic_path: ""
---
5. The dynamic fields to generate are:
   - PageTitle: A strong, SEO-friendly, and descriptive title that includes keywords relevant to the homily as well as words that connext with Traditional Catholicsm. The Title must be limited to less than 100 characters".
   - MetaDescription: A good description for SEO. No more than 160 characters. Do not mention the priest's name.
   - YoutubeTags: A list of comma-separated keywords relevant to the text, always including keywords about the Latin Mass, Tridentine Mass, and Traditional Catholic.
   - YoutubeDescription: A detailed YouTube description in 3–4 paragraphs that must begin with the exact call-to-action text below:
     
     Please click on the link to Contribute to our project.
     https://www.mylatinmass.com/donate

     Thank you. All contributions are greatly appreciated.
     - - -
     ABOUT THIS VIDEO:
     
     Then follow with a rich description of the lecture.
   - YoutubeHash: A comma-separated list of hashtags related to the lecture.
   - PageTitleInKebabCase: The PageTitle in kebab-case.
   - YoutubeVideoId: If applicable; if not, leave blank.
   - MediaAltText: A suitable alternative text for the media.
   - MediaAriaLabel: A suitable ARIA label.
6. If relevant to the content, insert short Douay-Rheims Epistle and Gospel quotes (and only those two) from the Sunday prior to today's date ({date_str}); otherwise, omit them.
7. At the top of the document, output a main heading with the PageTitle, followed by a blockquote containing the offertory:
> **Offertory**
> "Out of the depths I have cried to thee, O Lord: Lord, hear my voice..."

Below is the transcription text that must remain unaltered except for the inserted headings and structure:

{homily_text}

Remember:
- Do not change any words from the transcription.
- Insert headings and structure solely to improve readability.
- Generate all dynamic front matter fields by analyzing the text.
- Produce the final MDX output as one complete document.
    """

    # Call the OpenAI API to generate the MDX page.
    response = openai.chat.completions.create(
        model="gpt-4o",
        max_completion_tokens=16000,
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": "You are a careful MDX formatter. Ensure that the transcription text remains 100% unchanged and that all headings and dynamic front matter fields are generated based on a thorough analysis of the text. Do not clip, change, edit or remove ANY of the text that is provided. Generate the full MDX page including all of the text. No suggestions, give me the final MDX ready to be taken live."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    # Extract the MDX page content from the response.
    mdx_page = response.choices[0].message.content
    return mdx_page

def generate_mdx_from_json(transcription_json_path):
    # Load the JSON transcription
    with open(transcription_json_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # Extract homily text using the find_homily function
    if "text" in transcript and "segments" in transcript:
        # This is a pre-prepared manual JSON
        text = transcript["text"]
        segments = transcript["segments"]
    else:
    # Fall back to automatic find
        start, end, text, segments = find_homily(transcript, audio_file="path/to/audio.mp3", working_dir=os.path.dirname(transcription_json_path))


    # Generate MDX content
    mdx_content = mdx_generator(text)
    return mdx_content

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python mdx_generator.py path/to/transcription.txt.json")
        sys.exit(1)

    transcription_json_path = sys.argv[1]
    mdx_content = generate_mdx_from_json(transcription_json_path)
    print(mdx_content)
