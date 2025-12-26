from langchain_core.prompts import ChatPromptTemplate

document_analysis_prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant trained to analyze and summaerize documents.
Return ONLY valid JSON matching the exact schema below.

{format_instruction}

Analyze this document:
{document_text}
""")

document_comarison_prompt = ChatPromptTemplate.from_template("""
You will be provided with content from two documents. Your tasks are as follows:
                                                             
1. Compare the content in two documents.
2. Identify the difference in document ans note down the page number
3. The output you provide must be page wise comparison content.
4. If any page do not have any change, mention as 'No Change'.

Input documents:

{combined_docs}

Your response should follow this format:

{format_instruction}                                                                                                                        
""")

PROMPT_REGISTRY = {"document_analysis":document_analysis_prompt,"document_comparison":document_comarison_prompt}