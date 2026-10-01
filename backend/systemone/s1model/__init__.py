"""The dedicated local System One model.

A non-generative cross-encoder that answers the Jev contract
(:mod:`systemone.s1.contract`). Every candidate answer of every question is
turned into a hypothesis ("<instructions> Answer: <option>.") and scored
against the state in one forward pass; a question's answer distribution is
the softmax of its options' scores, divided by a temperature fitted on held
out data so the probabilities are calibrated. Nothing is generated, so the
same request always produces the same answer on the same hardware.

* :mod:`.model` - load, score and answer (inference)
* :mod:`.train` - fine-tune on labelled or distilled decisions, fit temperatures
* :mod:`.data` - build training data (the phishing suite for now)
* :mod:`.serve` - HTTP server speaking Jev's ``POST /v1/systemone``

Requires ``torch`` and ``transformers`` (installed in the ``s1`` image only).
"""
