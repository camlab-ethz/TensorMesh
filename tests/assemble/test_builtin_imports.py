"""Import-compatibility tests for built-in assembler hierarchy."""

from __future__ import annotations


def test_top_level_assemble_imports_remain_compatible():
    from tensormesh.assemble import (
        ContactAssembler,
        DruckerPragerPlasticity,
        J2Plasticity,
        LaplaceElementAssembler,
        LinearElasticityElementAssembler,
        MassElementAssembler,
        NeoHookeanModel,
        const_node_assembler,
        func_node_assembler,
    )

    assert LaplaceElementAssembler is not None
    assert MassElementAssembler is not None
    assert LinearElasticityElementAssembler is not None
    assert NeoHookeanModel is not None
    assert J2Plasticity is not None
    assert DruckerPragerPlasticity is not None
    assert ContactAssembler is not None
    assert const_node_assembler is not None
    assert func_node_assembler is not None


def test_builtin_package_imports_remain_compatible():
    from tensormesh.assemble.builtin import J2Plasticity as J2FromBuiltin
    from tensormesh.assemble.builtin import (
        DruckerPragerPlasticity as DPFromBuiltin,
    )
    from tensormesh.assemble.builtin.solid import J2Plasticity as J2FromSolid
    from tensormesh.assemble.builtin.solid import (
        DruckerPragerPlasticity as DPFromSolid,
    )

    assert J2FromBuiltin is J2FromSolid
    assert DPFromBuiltin is DPFromSolid


def test_category_imports_work():
    from tensormesh.assemble.builtin.fluid import (
        LaplaceElementAssembler,
        MassElementAssembler,
    )
    from tensormesh.assemble.builtin.solid import (
        ContactAssembler,
        DruckerPragerPlasticity,
        J2Plasticity,
        LinearElasticityElementAssembler,
        NeoHookeanModel,
    )
    import tensormesh.assemble.builtin.electromagnetic as electromagnetic

    assert LaplaceElementAssembler is not None
    assert MassElementAssembler is not None
    assert LinearElasticityElementAssembler is not None
    assert NeoHookeanModel is not None
    assert J2Plasticity is not None
    assert DruckerPragerPlasticity is not None
    assert ContactAssembler is not None
    assert electromagnetic.__all__ == []
