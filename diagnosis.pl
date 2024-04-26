:- dynamic symptom/2.
:- dynamic risk_factor/2.
:- dynamic diagnostic_criteria/3.

:- discontiguous symptom/2.
:- discontiguous risk_factor/2.
:- discontiguous diagnostic_criteria/3.

% Extended Patient-specific Symptoms
symptom(_, frequent_urination).
symptom(_, increased_thirst).
symptom(_, unexplained_weight_loss).
symptom(_, extreme_hunger).
symptom(_, blurred_vision).
symptom(_, slow_healing_wounds).
symptom(_, tingling_hands_feet).

% Extended Patient-specific Risk Factors
risk_factor(_, obesity).
risk_factor(_, sedentary_lifestyle).
risk_factor(_, family_history_diabetes).
risk_factor(_, high_blood_pressure).

diagnostic_criteria(Patient, blood_glucose, Level) :-
    nonvar(Level),
    nonvar(Patient), % This is to use the Patient variable and avoid the singleton warning
    Level >= 126.

diagnostic_criteria(Patient, hba1c, Level) :-
    nonvar(Level),
    nonvar(Patient), % Use the Patient variable
    Level >= 6.5.

diagnostic_criteria(Patient, fasting_glucose, Level) :-
    nonvar(Level),
    nonvar(Patient), % Use the Patient variable
    Level >= 100, Level < 126.

% Handle multiple symptoms more dynamically
add_symptom(Patient, Symptom) :-
    assertz(symptom(Patient, Symptom)).
% Incorporating age into the diagnostic criteria
diagnostic_criteria(Patient, age, Age) :-
    nonvar(Age),
    nonvar(Patient),
    assertz(age(Patient, Age)).

% Detailed Rules for Diagnosis
% Diabetes diagnosis

% Diabetes diagnosis incorporating age
diagnosis(Patient, diabetes) :-
    age(Patient, Age), Age >= 45,  % Considering age 45+ as a risk factor
    symptom(Patient, frequent_urination),
    symptom(Patient, increased_thirst),
    risk_factor(Patient, obesity),
    diagnostic_criteria(Patient, blood_glucose, LevelBG),
    LevelBG >= 126.

% Pre-diabetes diagnosis incorporating age as additional info
diagnosis(Patient, pre_diabetes) :-
    age(Patient, Age), Age >= 35,  % Lowering age threshold for pre-diabetes
    symptom(Patient, increased_thirst),
    symptom(Patient, frequent_urination),
    diagnostic_criteria(Patient, hba1c, LevelHb),
    LevelHb >= 5.7, LevelHb < 6.5.

% No diabetes
diagnosis(Patient, non_diabetic) :-
    \+ diagnosis(Patient, diabetes),
    \+ diagnosis(Patient, pre_diabetes).


% Adding patient details including age
add_patient_details(Patient, Age, Symptom, Risk, DiagnosticCriterion, Level) :-
    assertz(age(Patient, Age)),
    assertz(symptom(Patient, Symptom)),
    assertz(risk_factor(Patient, Risk)),
    assertz(diagnostic_criteria(Patient, DiagnosticCriterion, Level)),
    diagnosis(Patient, Diagnosis),
    format('~w, aged ~w, is diagnosed with ~w.~n', [Patient, Age, Diagnosis]).

% Example usage


% Usage
add_patient_symptoms(Patient, Symptoms) :-
    maplist(add_symptom(Patient), Symptoms).

% Example usage
?- add_patient_details('john_doe', 48, 'increased_thirst', 'obesity', 'blood_glucose', 130).

