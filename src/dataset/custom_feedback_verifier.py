class Feedback:
    def __init__(self, env, action):
        self.env = env
        self.action = action
        # access description properties (e.g. desc) and methods (e.g. verify())
        # in sequence tasks using instrs.instr_a / instrs.instr_b
        # check if BeforeInstr / AfterInstr sequence tasks include nested AndInstr's
        # and use instrs.instr_a.instr_a ... if this is the case

        # PutNext instructions don't have an attribute desc - use desc_move and desc_fixed

    def verify_feedback(self):
        if not self.action:
            return ""
        else:
            return "feedback"
