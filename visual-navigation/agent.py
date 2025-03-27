from typing import Dict, Union


def checks_terminate_message(msg):
    if isinstance(msg, str):
        return msg.find("TERMINATE") > -1
    elif isinstance(msg, dict) and 'content' in msg:
        return msg['content'].find("TERMINATE") > -1
    else:
        print(type(msg), msg)
        raise NotImplementedError


class VisualNavigationUserAgent():
    
    def __init__(
        self,
        prompt_generator, 
        executor,
        is_termination_msg = checks_terminate_message,
    ):
        super().__init__()
        self.prompt_generator = prompt_generator
        self.executor = executor
        self.is_termination_msg = is_termination_msg

    def receive(
        self,
        message: Union[Dict, str],
    ):
        """Process the received messages
        """
        # parsing the code component, if there is one
        # if parsing succeeds, then execute the code component
        if self.executor:
            # go to execution stage if there is an executor module
            results = self.executor.execute(message)
            output = results['message']
            file_pths = results['file_pths']
            exit_status = results['status']
            if checks_terminate_message(output):
                return {'content': output, 'terminator': True}
            reply = self.prompt_generator.get_exec_feedback(exit_status, output, file_pths)
            return {'content': reply, 'terminator': False}
            
    def generate_init_message(self, task_path: str):
        content = self.prompt_generator.initial_prompt(task_path)
        return content

    def initiate_chat(self, task_path: str):
        initial_message = self.generate_init_message(task_path)
        return initial_message


