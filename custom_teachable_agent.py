from autogen import ConversableAgent

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

class CustomTeachableAgent(ConversableAgent):

    def consider_memo_storage(self, comment):
        """Decides whether to store something from one user comment in the DB."""
        # Check for broad personal information
        response = self.analyze(
            comment,
            "Does the TEXT include any details that are important for a mental health coach to know a user such as the user's relationships, past relationships, work, hobbies, coworkers, or memories? Respond with yes or no.",
        )
        if "yes" in response.lower():
            # Extract all relevant information.
            context_info = self.analyze(
                comment,
                "Extract all details from the TEXT that are important for a mental health coach, including any mentions of people, relationships, work, hobbies, memories, or other relevant details.",
            )
            if context_info.strip():
                # Formulate a question this information could answer.
                question = self.analyze(
                    comment,
                    "If someone asked for a summary of this user's relationships, work, hobbies, or memories based on the TEXT, what question would they be asking? Provide the question only.",
                )
                # Store the CRM information as a memo.
                if self.verbosity >= 1:
                    print(colored("\nREMEMBER THIS CONTEXT INFORMATION", "light_yellow"))
                self.memo_store.add_input_output_pair(question, context_info)

    def consider_memo_retrieval(self, comment):
        """Decides whether to retrieve memos from the DB, and add them to the chat context."""
        # Directly use the user comment for memo retrieval.
        memo_list = self.retrieve_relevant_memos(comment)

        # Additional context-specific check.
        response = self.analyze(
            comment,
            "Does the TEXT request information on a user's specific relationship, work, hobby, or memory? Answer with yes or no.",
        )
        if "yes" in response.lower():
            # Retrieve relevant context memos.
            context_query = "What are the user's relevant relationships, work, hobbies, or memories needs based on previous interactions?"
            memo_list.extend(self.retrieve_relevant_memos(context_query))

        # De-duplicate the memo list.
        memo_list = list(set(memo_list))

        # Append the memos to the last user message.
        return comment + self.concatenate_memo_texts(memo_list)