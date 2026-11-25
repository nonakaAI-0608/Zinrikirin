"""Bounded in-memory undo history for editable clip states."""
from copy import deepcopy

class History:
    def __init__(self, limit=100):
        self.limit=limit
        self.clear()

    def clear(self):
        self.past=[]; self.future=[]; self.token=None

    def boundary(self):
        self.token=None

    def record(self,state,token=None):
        if token is None or token!=self.token:
            if not self.past or self.past[-1]!=state:
                self.past.append(deepcopy(state))
                self.past=self.past[-self.limit:]
        self.future.clear()
        self.token=token

    def undo(self,current):
        self.boundary()
        if not self.past:return None
        self.future.append(deepcopy(current))
        return self.past.pop()

    def redo(self,current):
        self.boundary()
        if not self.future:return None
        self.past.append(deepcopy(current))
        return self.future.pop()
