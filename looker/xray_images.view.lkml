view: patients {
    sql_table_name: patients ;;
    dimension: name {
        type: string
    }
    dimension: age {
        type: number
    }
    dimension: gender {
        type: string
    }
}

view: xray_images {
    sql_table_name: xray_images ;;
    dimension: diagnosis {
        type: string
    }
    measure: confidence {
        type: number
    }
    dimension: date_uploaded {
        type: date
    }
}
