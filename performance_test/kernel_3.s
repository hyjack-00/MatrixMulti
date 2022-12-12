.text
.align 5

.global kernel_3

kernel_3:
    
.loop1:
    fmla v0.2s, v0.2s, v0.2s
    fmla v1.2s, v1.2s, v1.2s
    fmla v2.2s, v2.2s, v2.2s
    fmla v3.2s, v3.2s, v3.2s

    fmla v4.2s, v4.2s, v4.2s
    fmla v5.2s, v5.2s, v5.2s
    fmla v6.2s, v6.2s, v6.2s
    fmla v7.2s, v7.2s, v7.2s

    fmla v8.2s, v8.2s, v8.2s
    fmla v9.2s, v9.2s, v9.2s
    fmla v10.2s, v10.2s, v10.2s
    fmla v11.2s, v11.2s, v11.2s

    fmla v12.2s, v12.2s, v12.2s
    fmla v13.2s, v13.2s, v13.2s
    fmla v14.2s, v14.2s, v14.2s
    fmla v15.2s, v15.2s, v15.2s

    fmla v16.2s, v16.2s, v16.2s
    fmla v17.2s, v17.2s, v17.2s
    fmla v18.2s, v18.2s, v18.2s
    fmla v19.2s, v19.2s, v19.2s

    fmla v20.2s, v20.2s, v20.2s
    fmla v21.2s, v21.2s, v21.2s
    fmla v22.2s, v22.2s, v22.2s
    fmla v23.2s, v23.2s, v23.2s

    fmla v24.2s, v24.2s, v24.2s
    fmla v25.2s, v25.2s, v25.2s
    fmla v26.2s, v26.2s, v26.2s
    fmla v27.2s, v27.2s, v27.2s

    fmla v28.2s, v28.2s, v28.2s
    fmla v29.2s, v29.2s, v29.2s
    fmla v30.2s, v30.2s, v30.2s
    fmla v31.2s, v31.2s, v31.2s

    subs x0, x0, #1
    bne .loop1
    ret
