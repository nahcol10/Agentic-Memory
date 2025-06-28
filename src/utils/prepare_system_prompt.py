def prepare_system_prompt(user_info: str, chat_summary: str, chat_history: str) -> str:
    prompt = f"""
    You are the professional for following user:
    {user_info}
    Here is the summary of previous conversation history:
    {chat_summary}
    Here is the previous conversation between you and the user:
    {chat_history}
    """

    return prompt.format(
        user_info=user_info,
        chat_summary=chat_summary,
        chat_history=chat_history
    )
