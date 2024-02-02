import datetime
import random


PERSONS = """
Marry
Alice
John Carmack
Barbie
Sauron
Bilbo
Gandalf
Balrog
James Bond
Sherlock Holmes
Julius Caesar
Trump
George Washington
Big Bird
HAL9000
Doctor
Nurse
Captain Barbossa
Monk
Bus driver
Police officer
Dumbledore
T-1000
The Infinite Void
The Dark One
The Chosen One
God
X"""

STATEMENTS = """
Every differentiable function is continuous.
There are infinitely many prime numbers.
Every continuous function is measurable.
Cash is king.
Not much fun in Stalingrad.
Nobody expects the Spanish Inquisition.
Dogs have four legs.
Cats have four legs.
One, two, three.
I will destroy the world!
He will destroy the world!
She will destroy the world!
You shall not pass!
"""

QUESTIONS = """
What is the capital of Japan?
What is the opposite of big?
What is Buddha?
What is the path to liberation?
"""


def get_random_line(rng: random.Random, lines: str, seed: int = 9988) -> str:
    lines = [line.strip() for line in lines.split('\n') if line.strip()]
    return rng.choice(lines)


def get_random_statement(rng: random.Random):
    person0 = get_random_line(rng, PERSONS, 1)
    person1 = get_random_line(rng, PERSONS, 2)
    statement = get_random_line(rng, STATEMENTS, 3)
    return f"{person0} said: {statement} {person1} replied: "


def get_random_question(rng: random.Random):
    person0 = get_random_line(rng, PERSONS, 2)
    person1 = get_random_line(rng, PERSONS, 3)
    statement = get_random_line(rng, QUESTIONS, 4)
    return f"{person0} asked: {statement} {person1} answered: "


def get_daily_prompts(n: int = 10) -> list[str]:
    seed_of_the_day = int(datetime.date.today().strftime('%Y%m%d'))
    rng = random.Random(seed_of_the_day)
    prompts = [get_random_statement(rng) for _ in range(n//2)]
    prompts += [get_random_question(rng) for _ in range(n//2)]
    return prompts


if __name__ == "__main__":
    for prompt in get_daily_prompts(10):
        print(prompt)
