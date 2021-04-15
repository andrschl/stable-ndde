cd(@__DIR__)
using Pkg; Pkg.activate("../../."); Pkg.instantiate(); using Revise
# using Pkg; Pkg.activate("../../../node_julia/."); Pkg.instantiate(); using Revise
using Flux, Zygote, DifferentialEquations, DiffEqSensitivity, DataStructures, ReverseDiff
using ModelingToolkit, Symbolics, Latexify
using Plots, StatsBase
using Random, Distributions
using Calculus
using CairoMakie, ColorSchemes
using LinearAlgebra
