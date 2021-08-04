library(tidyverse)
library(tidymodels)
library(shieldsthemes)
library(glue)
library(lubridate)

file_path = "C:/RScripts/Sliced/sliced-s01e10-playoffs-2"
dataset = read_csv(glue("{file_path}/train.csv"))


data = dataset %>% 
  mutate_if(is.character, str_trim) %>% # clean spaces on text columns
  mutate(age_at_out = as.Date(datetime) - date_of_birth, # age in days at outcome var,
         animal_type_fix = ifelse(!str_detect(animal_type, "Dog|Cat"), "Other", animal_type) %>% # convert animal to dog/cat/other
                            str_to_upper() %>% 
                            factor(),
         n_yrs = as.integer(age_at_out/365), # num years old
         neutered = ifelse(sex == "Male" & spay_neuter == "Fixed", 1, 0),
         qtr_of_decis = quarter(datetime),
         hour_of_decis = format(datetime, format = "%H:%M:%S") %>% 
                          str_extract("^(\\d{2})") %>% 
                          as.numeric(),
         biz_hours = ifelse(between(hour_of_decis, 8, 17), 1,0), 
         no_name = ifelse(is.na(name), 1, 0),
         mix_breed = ifelse(str_detect(breed, "Mix"), 1, 0),
         day_of_week = weekdays(datetime),
         pitbull = ifelse(str_detect(breed, "Pitbull"), 1, 0)) %>% 
  transform(age_at_out = as.numeric(age_at_out)) %>% 
  mutate_if(is.character, as.factor) %>% 
  select(-c(breed, color, name, animal_type, age_upon_outcome, date_of_birth))
  
data %>% 
  group_by(animal_type_fix, outcome_type) %>% 
  summarise(n())

data %>% 
  ggplot() +
  aes(x = age_at_out, fill = animal_type_fix) +
  geom_density(alpha = 0.5)

data %>% 
  group_by(qtr_of_decis, outcome_type) %>% 
  summarise(n = n()) %>% 
  ggplot() +
  aes(x = qtr_of_decis, y = n, group = outcome_type, fill = outcome_type) +
  geom_bar(stat = "identity") +
  facet_wrap(~outcome_type)

data %>% 
  group_by(outcome_type, biz_hours) %>% 
  summarise(n())

## Make Split -------------------------------------------------------------------------------------
set.seed(15)

split = initial_split(data, 0.66)
train = training(split)
test = testing(split)

## Make Model Spec --------------------------------------------------------------------------------

rec = train %>% 
  recipe(outcome_type ~ .) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  update_role(id, new_role = "ID") %>% 
  step_holiday(datetime) %>% 
  step_rm(datetime) %>% 
  step_
  prep()

model = rand_forest(trees = 1000, min_n = 5, mtry = 8) %>% 
  set_mode("classification") %>% 
  set_engine("ranger")

wf = workflow() %>% 
  add_model(model) %>% 
  add_recipe(rec)

fit_model = wf %>% fit(train)

## Try Test ---------------------------------------------------------------------------------------

fit_model %>% 
  predict(test) %>% 
  bind_cols(test %>% select(outcome_type)) %>% 
  group_by(`.pred_class`, outcome_type) %>% 
  summarise(n = n()) %>% 
  group_by(outcome_type) %>% 
  mutate(npct = n/sum(n))

fit_model %>% accuracy()

fit_model %>% 
  predict(test) %>% 
  bind_cols(test %>% select(outcome_type)) %>% 
  accuracy(`.pred_class`, outcome_type)

fit2 = wf %>% fit(data)

## Predict Test -----------------------------------------------------------------------------------

valid = read_csv(glue("{file_path}/test.csv")) %>% 
  mutate_if(is.character, str_trim) %>% # clean spaces on text columns
  mutate(age_at_out = as.Date(datetime) - date_of_birth, # age in days at outcome var,
         animal_type_fix = ifelse(!str_detect(animal_type, "Dog|Cat"), "Other", animal_type) %>% # convert animal to dog/cat/other
           str_to_upper() %>% 
           factor(),
         n_yrs = as.integer(age_at_out/365), # num years old
         neutered = ifelse(sex == "Male" & spay_neuter == "Fixed", 1, 0),
         qtr_of_decis = quarter(datetime),
         hour_of_decis = format(datetime, format = "%H:%M:%S") %>% 
           str_extract("^(\\d{2})") %>% 
           as.numeric(),
         biz_hours = ifelse(between(hour_of_decis, 8, 17), 1,0), 
         no_name = ifelse(is.na(name), 1, 0),
         mix_breed = ifelse(str_detect(breed, "Mix"), 1, 0),
         day_of_week = weekdays(datetime),
         pitbull = ifelse(str_detect(breed, "Pitbull"), 1, 0)) %>% 
  transform(age_at_out = as.numeric(age_at_out)) %>% 
  mutate_if(is.character, as.factor) %>% 
  select(-c(breed, color, name, animal_type, age_upon_outcome, date_of_birth))

fit2 %>% 
  predict(valid) %>% 
  bind_cols(valid %>% select(id)) %>% 
  rename("class" = 1) %>% 
  mutate(val = 1) %>% 
  pivot_wider(names_from = 'class', values_from = 'val') %>% 
  mutate(across(2:last_col(), ~replace_na(.x, 0))) %>% 
  select(id, adoption, `no outcome`, transfer) %>% 
  write_csv(glue("{file_path}/argeros_preds07.csv"))

