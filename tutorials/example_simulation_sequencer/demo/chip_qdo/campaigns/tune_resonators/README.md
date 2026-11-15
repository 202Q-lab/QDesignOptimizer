# Tune resonators

**Question.** Can R1 and R2 both reach their frequency and linewidth
targets with the coupling lengths alone?

This campaign has no `design_variables.json` of its own, so its studies
read and write the chip's own store. A sub-campaign made with
`campaign new tune_resonators/<name>` gets its own copy of that store,
and writes nothing outside it until its values are promoted.

## Log

- (date: what was run, under which study ids, and what it showed)

## Conclusion

(the answer, and whether its values were promoted)
