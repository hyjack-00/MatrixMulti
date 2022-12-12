.text
.align 5

.global kernel_1

kernel_1:
    
.loop1:
    fmla v0.4s, v0.4s, v0.4s
    fmla v1.4s, v1.4s, v1.4s
    fmla v2.4s, v2.4s, v2.4s
    fmla v3.4s, v3.4s, v3.4s

    fmla v4.4s, v4.4s, v4.4s
    fmla v5.4s, v5.4s, v5.4s
    fmla v6.4s, v6.4s, v6.4s
    fmla v7.4s, v7.4s, v7.4s

    fmla v8.4s, v8.4s, v8.4s
    fmla v9.4s, v9.4s, v9.4s
    fmla v10.4s, v10.4s, v10.4s
    fmla v11.4s, v11.4s, v11.4s

    fmla v12.4s, v12.4s, v12.4s
    fmla v13.4s, v13.4s, v13.4s
    fmla v14.4s, v14.4s, v14.4s
    fmla v15.4s, v15.4s, v15.4s

    fmla v16.4s, v16.4s, v16.4s
    fmla v17.4s, v17.4s, v17.4s
    fmla v18.4s, v18.4s, v18.4s
    fmla v19.4s, v19.4s, v19.4s

    fmla v20.4s, v20.4s, v20.4s
    fmla v21.4s, v21.4s, v21.4s
    fmla v22.4s, v22.4s, v22.4s
    fmla v23.4s, v23.4s, v23.4s

    fmla v24.4s, v24.4s, v24.4s
    fmla v25.4s, v25.4s, v25.4s
    fmla v26.4s, v26.4s, v26.4s
    fmla v27.4s, v27.4s, v27.4s

    fmla v28.4s, v28.4s, v28.4s
    fmla v29.4s, v29.4s, v29.4s
    fmla v30.4s, v30.4s, v30.4s
    fmla v31.4s, v31.4s, v31.4s

    subs x0, x0, #1
    bne .loop1
    ret
