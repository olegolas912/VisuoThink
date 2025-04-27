from typing import Dict, Union


def checks_terminate_message(msg):
    if isinstance(msg, str):
        return msg.find("TERMINATE") > -1
    elif isinstance(msg, dict) and 'content' in msg:
        return msg['content'].find("TERMINATE") > -1
    else:
        print(type(msg), msg)
        raise NotImplementedError


def checks_aux_end_message(msg):
    if isinstance(msg, str):
        return msg.find("AUX-END") > -1
    elif isinstance(msg, dict) and 'content' in msg:
        return msg['content'].find("AUX-END") > -1
    else:
        print(type(msg), msg)
        raise NotImplementedError


class GeoProUserAgent():
    
    def __init__(
        self,
        prompt_generator, 
        parser,
        executor,
        is_termination_msg = checks_terminate_message,
        step_aux = False
    ):
        super().__init__()
        self.prompt_generator = prompt_generator
        self.parser = parser
        self.executor = executor
        self.is_termination_msg = is_termination_msg
        self.aux_end = False if step_aux else None # whether the auxiliary lines are ended
        
    def receive(
        self,
        message: Union[Dict, str],
    ):
        """Process the received messages
        """
        # parsing the code component, if there is one
        parsed_results = self.parser.parse(message)
        parsed_content = parsed_results['content']
        parsed_status = parsed_results['status']
        parsed_error_message = parsed_results['message']
        parsed_error_code = parsed_results['error_code']
        
        # if TERMINATION message, then return
        if not parsed_status and self.is_termination_msg(message):
            return
        
        # if aux_end message, then set aux_end to True
        if self.aux_end is not None and checks_aux_end_message(message):
            self.aux_end = True
        if not parsed_status and checks_aux_end_message(message): # 只有 AUX-END
            self.aux_end = True
            reply = self.prompt_generator.get_exec_feedback(0, '', aux_end=True)
            return reply

        # if parsing fails
        if not parsed_status:    
            reply = self.prompt_generator.get_parsing_feedback(parsed_error_message, parsed_error_code)
            return reply
        
        # if parsing succeeds, then execute the code component
        if self.executor:
            # go to execution stage if there is an executor module
            exit_code, output, file_paths = self.executor.execute(parsed_content)
            if self.aux_end is not None:
                reply = self.prompt_generator.get_exec_feedback(exit_code, output, aux_end=self.aux_end)
            else:
                reply = self.prompt_generator.get_exec_feedback(exit_code, output)
            return reply
            
    def generate_init_message(self, query, n_image):
        content = self.prompt_generator.initial_prompt(query, n_image)
        return content

    def initiate_chat(self, message):
        initial_message = self.generate_init_message(message, 0)
        return initial_message