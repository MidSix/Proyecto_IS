This is a file that enumerate some rules that we all agreed on
or are mandatory by the teacher.
that are usefull to maintain the consistency throughout all the code

* The code comentaries should be in English

* Every commit message should follow the conventional commit style
(https://www.conventionalcommits.org/en/v1.0.0/) -> reference link

* We must follow PEP8(Python Enhancement Proposal) this means:
for every function < 79 columns and < 40 lines of code.
 How to follow this? "Under 30 - Function Length Checker & Line Counter"
-> extension that counts how many lines of code has a selected
block of code. Using the command ">Set Function Length Limit"
you can set 40 lines instead of 30. Now to see if any line of code
surpass 79 columns/characters, you can set a vertical ruler,
here's a link that teaches you how to do so:
https://www.youtube.com/shorts/HPKBkJPs2DM
It is also recommended but not for PEP 8(that's why isn't mandatory)
that each module has less than 400-500 lines of code, having more
could be a sympton of a god module that don't satisfy the single
purpose only principle from PEP 8 and even mix frontend/backend logic.
Naming conventions:
snake_case -> functions, methods, attributes, modules
, UPPERCASE -> constants
, CapWords  -> classes

-More rules will be added soon-