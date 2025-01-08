from collections import defaultdict
import numpy as np


class ExpenseDistributor:
    """
    Class to calculate group balances from expenses (like Tricount or Splitwise). Here, the benefit lies in using
    weights assigned to each participant, by which the expense is multiplied to adapt to positions within the lab.
    Hence, the algorithm takes into account that students should pay less, whereas postdocs might want to pay more.
    Use cases might be: Christmas parties, ...
    Workflow:
    1. Add participants with according weights to the class
    2. Add expenses and respective cohort
    3. Calculate group balance
    4. Minimize number of transactions
    """
    def __init__(self):
        self.expenses = []
        self.participants = set()
        self.weights = {}  # Dictionary to store participant weights
        self.total_weight = float()

    def add_participant(self, name, weight=1.0):
        """
        Adds a participant with an optional weight.

        Args:
          name: The name of the participant.
          weight: The weight of the participant (default is 1).
        """
        self.participants.add(name)
        self.weights[name] = weight

    def add_expense(self, payer, amount, participants):
        """
        Adds a new expense to the Tricount.

        Args:
          payer: The person who paid for the expense.
          amount: The total amount of the expense.
          participants: A list of participants who shared the expense.
        """
        self.expenses.append({
            'payer': payer,
            'amount': amount,
            'participants': participants
        })
        self.participants.update(participants)

    def calculate_balances(self):
        """
        Calculates the balances for each participant, considering their weights.

        Returns:
          A dictionary where keys are participant names and values are their balances.
        """
        balances = defaultdict(float)
        for expense in self.expenses:
            payer = expense['payer']
            amount = expense['amount']
            self.total_weight = sum(self.weights[p] for p in expense['participants'])
            self.check_total_weight(expense)
            balances[payer] += amount  # Credit the payer with the full amount
            for participant in expense['participants']:
                share = amount * (self.weights[participant] / self.total_weight)  # Weighted share
                balances[participant] -= share

        return balances

    def check_total_weight(self, expense):
        if round(self.total_weight, 2) != len(expense["participants"]):
            raise ValueError("Sum of weights is not equal to number of participants!\n"
                             f"Sum of weights: {self.total_weight}\n"
                             f"Length of participants: {len(expense['participants'])}")

    def simplify_debts(self):
        """
        Simplifies the debts between participants to minimize transactions.

        Returns:
          A list of tuples, where each tuple represents a transaction (payer, receiver, amount).
        """
        balances = self.calculate_balances()
        positive_balances = [
            (person, balance) for person, balance in balances.items() if balance > 0
        ]
        negative_balances = [
            (person, balance) for person, balance in balances.items() if balance < 0
        ]

        transactions = []
        while positive_balances and negative_balances:
            payer, payer_balance = positive_balances.pop(0)
            receiver, receiver_balance = negative_balances.pop(0)
            amount = min(payer_balance, -receiver_balance)
            transactions.append((payer, receiver, amount))
            payer_balance -= amount
            receiver_balance += amount
            if payer_balance > 0:
                positive_balances.insert(0, (payer, payer_balance))
            if receiver_balance < 0:
                negative_balances.insert(0, (receiver, receiver_balance))

        return np.array(transactions, dtype="object")


if __name__ == "__main__":
    # instantiate
    ed = ExpenseDistributor()
    # Participants for two events
    dinner_participants = ["Andreja", "Anne", "Hannah", "Jessica", "Justus", "Martin",
                            "Max", "Merle", "Mohsen", "Frauke", "Sarah", "Malte", "Clara", "Jonas"]
    bowling_participants = ["Andreja", "Anne", "Hananja", "Hannah", "Jessica", "Justus", "Marius", "Martin",
                            "Max", "Merle", "Mohsen"]
    all_participants = list(set(dinner_participants) | set(bowling_participants))

    for participant in all_participants:
        if participant in ["Hannah", "Marius", "Hananja"]:
            ed.add_participant(participant, weight=0.6)
        elif participant in ["Anne", "Merle", "Jessica", "Andreja", "Max", "Justus", "Clara"]:
            ed.add_participant(participant, weight=0.85)
        elif participant in ["Malte", "Sarah", "Mohsen", "Frauke", "Martin"]:
            ed.add_participant(participant, weight=1.45)
        elif participant in ["Jonas"]:
            ed.add_participant(participant, weight=1)
    # add expenses
    ed.add_expense("Max", 177, participants=all_participants)
    ed.add_expense("Jonas", 247, participants=all_participants)
    # get balances
    balances = ed.calculate_balances()
    # reduce number of transactions
    transactions = ed.simplify_debts()