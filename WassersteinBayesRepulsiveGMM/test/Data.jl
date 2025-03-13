module Data 

using CSV
using DataFrames

export load_data

function load_data(datafile) 
    df = "./data/$(datafile).csv" |> CSV.File |> DataFrame
    
    df_ = nothing
    if datafile == "faithful_data.csv"
        df_ = df[!, 2:3]
    else 
        df_ = df[!, 1:2]
    end 

    return df_ |> Matrix |> transpose 
end 

end 